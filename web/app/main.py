"""API de la webapp.

Le contrat tient en quatre routes, et son point non négociable est que
le **navigateur fait tout le travail média**. Le serveur ne reçoit que
des fenêtres audio de quelques Mo : le VPS n'a ni le stockage pour les
sources ni le CPU pour les encoder.

Deuxième contrainte de forme : Cloudflare coupe les requêtes vers
100 s, or transcrire 30 minutes d'audio prend souvent plus. L'envoi
d'une fenêtre répond donc **202 immédiatement** et la transcription
part en tâche de fond ; le navigateur suit l'avancement par ``GET``.
"""

from __future__ import annotations

import asyncio
import json
import html
import logging
import os
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi import Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cloud_transcription import (
    enrich_transcript_via_gemini,
    CLOUD_TRANSCRIPTION_MODELS,
    CloudChunkResult,
    CloudTranscriptionError,
    canonical_cloud_model_id,
    chunk_seconds_for_model,
    estimate_cloud_cost,
    merge_chunk_results,
    plan_audio_chunks,
    provider_for_model,
)

from .auth import AccessVerifier, AuthError
from .chatter import ChatterError, composer
from .coffre import Coffre, CoffreIndisponible
from .db import Database
from .odoo import OdooGateway, OdooUnavailable
from .secrets import GeminiKey, SecretError
from .identite import IdentiteGoogle
from .recovery import register_recovery_routes
from .stockage import MORCEAU, PORTEE as PORTEE_DRIVE, DriveStockage, StockageIndisponible
from .enqueteur import MODELE_ENQUETE, Conclusion, enqueter, enqueter_confirme
from .sonde import FENETRE_SECONDES, fournisseur, identifier
from .settings import Settings
from .terms import replace_term
from .transcription import context_for_chunk, transcribe_chunk, transcript_text

log = logging.getLogger("ekovideo.web")

# Profil d'upload imposé au navigateur — un seul endroit, parce que le
# navigateur obéit au serveur plutôt que de décider.
#
# MP3 64 kbps mono 16 kHz, soit **exactement** ce que produit
# `build_cloud_audio_cmd` aujourd'hui. Deux raisons. Gemini documente
# ses formats audio (wav, mp3, aiff, aac, ogg, flac) et l'Opus n'y
# figure pas : l'accepterait-il en pratique ? je n'en sais rien, et le
# vérifier demande une vraie clé. Et à format identique, la
# transcription issue du navigateur se compare trait pour trait à celle
# de l'app macOS — c'est la vérification du jalon.
#
# L'Opus reste l'optimisation visée : ~10,7 Mo/heure mesuré en M0
# contre ~28 en MP3. Le jour où on le valide contre l'API, c'est cette
# constante qui change, et rien d'autre.
AUDIO_PROFILE = {
    "codec": "mp3",
    "container": "mp3",
    "sample_rate": 16000,
    "channels": 1,
    "bitrate": 64000,
}

# L'extension compte : `_audio_mime` (cloud_transcription) en déduit le
# type déclaré à Gemini, et tout ce qui n'est pas .mp3 part en audio/wav.
CHUNK_SUFFIX = ".mp3"

# Les appels sont du réseau, pas du CPU : une concurrence basse suffit
# et protège les 256 Mo du conteneur.
MAX_CONCURRENT_CHUNKS = 2

# Une fenêtre de 30 min pèse ~14 Mo en MP3 64 kbps (~5 Mo en Opus). La
# marge couvre un réglage plus généreux sans jamais approcher les 100 Mo
# du tunnel.
MAX_CHUNK_BYTES = 60 * 1024 * 1024

# Écrit par l'étape « manifeste » du Dockerfile, à côté du paquet.
MANIFESTE_SERVI = Path(__file__).resolve().parent.parent / "ekonum-app.json"
# Cinq minutes en MP3 64 kbit/s pèsent 2,4 Mo : la marge permet une
# fenêtre un peu plus large, pas le dépôt d'une réunion entière.
MAX_SONDE_BYTES = 8 * 1024 * 1024

# Le ré-enrichissement ne relit que du texte : le modèle le moins cher
# du catalogue suffit, et c'est ce qui rend la correction d'une liaison
# indolore.
MODELE_ENRICHISSEMENT = "gemini-3.1-flash-lite"

# Ce qui se retraite : une réunion qui n'a jamais eu de transcription.
# Une en cours appartient au navigateur qui l'encode. Une interrompue qui
# n'est pas à la corbeille, c'est un retraitement arrêté en route.
REPROCESSABLE = {"recovered", "erreur", "cancelled"}

# Du plus sûr au moins sûr. Lier sans demander à partir de « probable »
# est un réglage possible ; ce n'est pas le défaut.
ECHELLE = ["certaine", "probable", "incertaine"]


def liaison_sans_demander(confiance: str, seuil: str) -> bool:
    if seuil not in ECHELLE or confiance not in ECHELLE:
        return False
    return ECHELLE.index(confiance) <= ECHELLE.index(seuil)


class JobRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    duration_seconds: float = Field(gt=0)
    model: str = Field(min_length=1)
    language: str = "fr"
    context: dict[str, Any] = Field(default_factory=dict)
    # Quand la réunion a eu lieu — pas quand on l'a déposée. Proposée
    # d'après la date du fichier, corrigeable à la main.
    meeting_date: str | None = Field(default=None, max_length=40)
    # Retraiter une réunion qui n'a pas de transcription — récupérée ou
    # en échec — plutôt que d'en créer une seconde.
    reprocess_job_id: int | None = None


class ImportedSegment(BaseModel):
    start: float = 0.0
    end: float = 0.0
    speaker: str = ""
    text: str = ""


class ImportedJob(BaseModel):
    """Une réunion déjà transcrite, reprise depuis l'app macOS."""

    filename: str = Field(min_length=1, max_length=512)
    created_at: str = ""
    title: str = ""
    duration_seconds: float = 0.0
    model: str = ""
    transcript: str = ""
    speakers: dict[str, str] = Field(default_factory=dict)
    technical_terms: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    segments: list[ImportedSegment] = Field(default_factory=list)
    meeting_date: str = Field(default="", max_length=40)


class OdooCredentials(BaseModel):
    """Clé API Odoo **personnelle**.

    Pas de clé partagée : la note déposée dans le chatter porte
    l'identité du propriétaire de la clé. Avec une clé commune, tout
    serait signé du même compte et l'attribution — la raison d'être du
    chatter — disparaîtrait.
    """

    login: str = Field(min_length=1, max_length=254)
    api_key: str = Field(min_length=8, max_length=512)


class OdooPublication(BaseModel):
    model: str = Field(min_length=1, max_length=64)
    record_id: int = Field(gt=0)
    header: str = Field(default="", max_length=500)


class ContextPatch(BaseModel):
    """Édition partielle : un champ absent est laissé tel quel, pour qu'un
    formulaire qui n'affiche pas les termes ne les efface pas."""

    title: str | None = None
    speakers: dict[str, str] | None = None
    technical_terms: list[str] | None = None
    meeting_date: str | None = Field(default=None, max_length=40)


class TermReplacement(BaseModel):
    old: str = Field(min_length=1)
    new: str = Field(min_length=1)


class TokenRequest(BaseModel):
    name: str = Field(default="", max_length=120)


class VocabularyRecord(BaseModel):
    terms: list[str] = Field(default_factory=list)


class Reenrichissement(BaseModel):
    """Recoller un dossier Odoo au texte déjà transcrit."""

    model: str = Field(default="", max_length=64)
    record_id: int = Field(default=0, ge=0)


class ClientError(BaseModel):
    job_id: int | None = None
    stage: str = Field(default="", max_length=40)
    file_name: str = Field(default="", max_length=500)
    message: str = Field(default="", max_length=4000)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=2000)


class RejectedRecord(BaseModel):
    model: str = Field(max_length=64)
    id: int = Field(ge=0)
    name: str = Field(default="", max_length=300)


class GuidedInvestigation(BaseModel):
    """Relancer l'enquête quand aucune proposition ne convient : la
    personne explique où chercher, dans une conversation."""

    clues: dict[str, Any] = Field(default_factory=dict)
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    rejected: list[RejectedRecord] = Field(default_factory=list, max_length=20)
    moment: str = Field(default="", max_length=40)


class EnvoiVideo(BaseModel):
    taille: int = Field(gt=0, le=20 * 1024**3)
    type: str = Field(default="video/mp4", max_length=64)
    # Remplacer la vidéo déjà stockée — une récupérée qui n'avait jamais
    # été compressée, par exemple.
    remplacer: bool = False


class DemandeAppareil(BaseModel):
    """Ce qu'un appareil dit de lui en demandant à être enrôlé."""

    appareil: str = Field(default="", max_length=120)


class CodeAppareil(BaseModel):
    code_appareil: str = Field(min_length=8, max_length=128)


class FinalizeResponse(BaseModel):
    job_id: int
    title: str
    transcript: str
    speakers: dict[str, str]
    technical_terms: list[str]
    uncertain: list[dict[str, Any]] = Field(default_factory=list)
    cost_usd: float
    # Ce qu'il est advenu du dépôt automatique, quand il y en avait un.
    odoo: dict[str, Any] = Field(default_factory=dict)


def _pli(texte: str) -> str:
    """Minuscules sans accents : « Prévelis » et « prevelis » sont le même
    mot mal entendu."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texte.casefold())
        if unicodedata.category(c) != "Mn"
    ).strip()


def _designe(passage: str, mot: str) -> bool:
    """Le passage douteux est-il ce mot-là ? Seulement s'il ne contient
    que lui, à la ponctuation près : une phrase entière qui le cite peut
    douter d'autre chose."""
    return bool(mot.strip()) and _pli(passage).strip(" .,;:!?«»\"'") == _pli(mot)


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    gemini_key: GeminiKey | None = None,
    verifier: AccessVerifier | None = None,
    odoo_factory=None,
    stockage_factory=None,
    google_opener=None,
) -> FastAPI:
    config = settings or Settings.from_env()
    db = database or Database(config.db_path)
    keys = gemini_key or GeminiKey(
        url=config.broker_url,
        token=config.broker_token,
        item=config.broker_item,
        field=config.broker_field,
        static_key=config.dev_api_key,
    )
    access = verifier or AccessVerifier(config.access_team_domain, config.access_aud)

    etat_stockage: dict[str, Any] = {}

    def stockage():
        """Le stockage vidéo, ou ``StockageIndisponible``.

        Construit à la première demande et gardé : il porte le jeton du
        compte de service, qu'on ne renégocie pas à chaque morceau.
        """
        if stockage_factory is not None:
            return stockage_factory()
        if not stockage_configure():
            raise StockageIndisponible("Stockage vidéo non configuré.")
        if "instance" not in etat_stockage:
            etat_stockage["instance"] = DriveStockage(
                IdentiteGoogle(
                    chemin_cle=config.gcp_cle,
                    fournisseur=config.gcp_fournisseur,
                    compte=config.gcp_compte,
                    emetteur=config.public_url,
                    portee=PORTEE_DRIVE,
                ),
                config.video_dossier,
            )
        return etat_stockage["instance"]

    def stockage_configure() -> bool:
        return bool(config.gcp_fournisseur and config.gcp_compte and config.video_dossier)

    def stockage_disponible() -> bool:
        return stockage_factory is not None or stockage_configure()

    def effacer(job_id: int) -> None:
        """Supprime une réunion pour de bon, vidéo comprise.

        La vidéo d'abord : si Drive refuse, la réunion reste à la
        corbeille et on réessaiera, plutôt que de laisser un fichier
        orphelin que plus rien ne référence.
        """
        job = db.get_job(job_id) or {}
        if job.get("video_file_id"):
            stockage().supprimer(str(job["video_file_id"]))
        db.supprimer_job(job_id)
    coffre = Coffre(config.secret_key)

    def odoo_pour(owner_id: int) -> OdooGateway:
        """La passerelle Odoo **de cette personne**.

        Pas de clé de service partagée, même en lecture : une recherche
        faite sous un compte commun ignorerait les règles d'accès de la
        personne et lui montrerait des dossiers qui ne sont pas les
        siens.
        """
        if odoo_factory is not None:
            return odoo_factory(owner_id)
        login, chiffree = db.odoo_credentials(owner_id)
        return OdooGateway(
            url=config.odoo_url,
            database=config.odoo_database,
            login=login,
            api_key=coffre.dechiffrer(chiffree) if (login and chiffree) else "",
        )
    if not access.configured and not config.dev_mode:
        raise RuntimeError(
            "Cloudflare Access n'est pas configuré (EKOVIDEO_ACCESS_TEAM_DOMAIN "
            "et EKOVIDEO_ACCESS_AUD) et le mode développement est désactivé : "
            "refus de démarrer sans authentification."
        )

    config.chunk_dir.mkdir(parents=True, exist_ok=True)
    # Le client est construit par Vite ; en production l'image Docker
    # embarque `dist`. En développement on lance plutôt `npm run dev`,
    # qui sert le client et relaie /api ici.
    static_dir = Path(__file__).resolve().parent.parent / "client" / "dist"
    app = FastAPI(title="EkoVideo", version="1.0")
    app.state.settings = config
    app.state.db = db
    app.state.keys = keys
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
    # asyncio ne garde qu'une référence faible sur les tâches : sans ce
    # jeu, une transcription en cours peut être ramassée par le GC.
    app.state.tasks = set()

    # -- identité ------------------------------------------------------

    def current_user(request: Request) -> int:
        """Qui fait cette requête.

        Deux voies. Un humain arrive avec un jeton Cloudflare Access ; un
        appel machine — script, intégration Odoo, serveur MCP — arrive
        avec un jeton d'API en `Authorization: Bearer`. Les deux
        aboutissent au même identifiant d'utilisateur, donc l'attribution
        des coûts et le cloisonnement des traitements valent pareil dans
        les deux cas.

        Le jeton d'API est examiné en premier : il est explicite, alors
        qu'un jeton Access peut traîner dans un cookie et servir par
        accident.
        """
        entete = request.headers.get("Authorization", "")
        if entete.startswith("Bearer "):
            owner = db.owner_for_api_token(entete[7:])
            if owner is None:
                raise HTTPException(
                    status.HTTP_401_UNAUTHORIZED,
                    "Jeton d'API inconnu ou révoqué.",
                )
            return owner

        if config.dev_mode:
            return db.user_id_for_email(config.dev_user_email)
        token = request.headers.get("Cf-Access-Jwt-Assertion", "")
        try:
            email = access.email_from_token(token)
        except AuthError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
        return db.user_id_for_email(email)

    def human_user(request: Request) -> int:
        """Comme ci-dessus, mais refuse un jeton d'API.

        Garde la gestion des jetons hors de portée des jetons eux-mêmes :
        un jeton volé ne doit pas pouvoir s'en fabriquer d'autres, ni
        révoquer ceux des collègues.
        """
        if request.headers.get("Authorization", "").startswith("Bearer "):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "La gestion des jetons demande une connexion personnelle.",
            )
        return current_user(request)

    def owned_job(job_id: int, owner_id: int) -> dict[str, Any]:
        job = db.get_job(job_id)
        if job is None or int(job["owner_id"]) != owner_id:
            # Même réponse dans les deux cas : le job d'un collègue ne doit
            # pas être distinguable d'un job inexistant.
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Traitement introuvable.")
        return job

    # -- routes --------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Vivant : aucune dépendance vérifiée, exprès. Sinon une panne du
        broker ferait redémarrer un conteneur qui n'y est pour rien."""
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        """Utilisable : la base répond, et le broker rend la clé Gemini.

        Sans l'une ou l'autre, on ne peut ni lire la bibliothèque ni
        transcrire — le portail doit le voir, pas un « ok » de façade.
        """
        problemes: list[str] = []
        try:
            with db.connect() as conn:
                conn.execute("SELECT 1").fetchone()
        except Exception as exc:  # noqa: BLE001 — on rapporte, on ne lève pas
            problemes.append(f"base : {exc}")
        try:
            if not keys.get():
                problemes.append("broker : clé Gemini vide")
        except SecretError as exc:
            problemes.append(f"broker : {exc}")
        if problemes:
            return JSONResponse({"status": "indisponible", "problemes": problemes},
                                status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return {"status": "ok"}

    @app.get("/.well-known/ekonum-app.json")
    def manifeste_ekonum():
        """Le manifeste du portail Ekonum, complété au build (version,
        commit, date). Absent hors image — en développement, en test —,
        il répond 404 plutôt que d'inventer des champs de build."""
        chemin = Path(os.environ.get("EKOVIDEO_MANIFESTE", MANIFESTE_SERVI))
        if not chemin.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Manifeste non construit.")
        return JSONResponse(json.loads(chemin.read_text(encoding="utf-8")))

    @app.get("/bienvenue", response_class=HTMLResponse)
    def welcome_page() -> HTMLResponse:
        """La page à partager.

        Un lien collé dans Odoo ou un e-mail est prévisualisé par un robot
        qui n'a pas de session : sur « / », il ne voit que la page de
        connexion d'Access. Cette page-ci est ouverte (contournement Access,
        comme les points de santé), porte le titre, la description et
        l'image de l'aperçu, et renvoie aussitôt un humain vers l'app — où
        Access lui demandera de se connecter.
        """
        base = config.public_url.rstrip("/")
        titre = "transcript — vos réunions, transcrites et rangées dans Odoo"
        description = (
            "Déposez l’enregistrement : transcript le transcrit, retrouve le bon "
            "dossier Odoo et l’y dépose à votre nom."
        )
        return HTMLResponse(f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titre}</title>
<meta name="description" content="{description}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Ekonum">
<meta property="og:title" content="{titre}">
<meta property="og:description" content="{description}">
<meta property="og:url" content="{base}/bienvenue">
<meta property="og:image" content="{base}/bienvenue/apercu.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="transcript : vos réunions, transcrites et rangées dans Odoo">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="/marque/icone.svg">
<script>location.replace("/")</script>
</head><body style="font-family:sans-serif;padding:2rem">
<p><a href="/">Ouvrir transcript</a></p>
</body></html>""")

    @app.get("/")
    def index():
        page = static_dir / "index.html"
        if not page.exists():
            # Message franc plutôt qu'un 404 opaque : l'oubli le plus
            # probable est simplement de n'avoir pas construit le client.
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Client non construit : lancez « npm run build » dans web/client "
                "(ou « npm run dev » pour développer).",
            )
        return FileResponse(page)

    @app.post("/api/jobs", status_code=status.HTTP_201_CREATED)
    def create_job(payload: JobRequest, owner_id: int = Depends(current_user)) -> dict:
        # `cloud_model_entry` accepte volontairement un identifiant inconnu
        # — l'app macOS laisse saisir un modèle tout juste sorti — en le
        # facturant au tarif le plus cher. Ici le modèle vient d'une liste
        # fermée : une coquille doit être refusée, pas facturée au prix fort.
        model = canonical_cloud_model_id(payload.model)
        if model not in {entry["id"] for entry in CLOUD_TRANSCRIPTION_MODELS}:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Modèle inconnu : {payload.model!r}.",
            )

        # Garde-fou budget : avant le premier octet uploadé, pas après.
        estimate = estimate_cloud_cost(payload.duration_seconds, model)
        spent = db.month_spend_usd()
        if spent + estimate["cost_usd"] > config.monthly_budget_usd:
            raise HTTPException(
                status.HTTP_402_PAYMENT_REQUIRED,
                f"Ce traitement coûterait environ {estimate['cost_usd']:.2f} $US et "
                f"dépasserait le plafond mensuel de l'équipe "
                f"({spent:.2f} / {config.monthly_budget_usd:.2f} $US déjà engagés).",
            )

        windows = plan_audio_chunks(
            payload.duration_seconds, chunk_seconds_for_model(model)
        )
        if payload.reprocess_job_id is not None:
            job_id = payload.reprocess_job_id
            existant = owned_job(job_id, owner_id)
            if existant["deleted_at"]:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Traitement introuvable.")
            # Une réunion en cours n'est pas retraitée sous les pieds du
            # navigateur qui l'encode ; une transcrite, on la relance.
            if existant["status"] not in REPROCESSABLE or existant["transcript"]:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "Seule une réunion sans transcription — récupérée ou en échec — se retraite.",
                )
            db.reprocess_job(
                job_id,
                duration_seconds=payload.duration_seconds,
                model=model,
                language=payload.language,
                context={**payload.context, "reprocessed": True},
                chunks=windows,
            )
        else:
            job_id = db.create_job(
                owner_id=owner_id,
                filename=payload.filename,
                duration_seconds=payload.duration_seconds,
                model=model,
                language=payload.language,
                context=payload.context,
                chunks=windows,
            )
        if payload.meeting_date:
            db.set_meeting_date(job_id, _date_reunion(payload.meeting_date))
        # Le dossier choisi avant la transcription est retenu maintenant :
        # à la fin, le dépôt n'aura plus rien à demander.
        dossier = payload.context.get("odoo_record") or {}
        if dossier.get("model") and dossier.get("record_id"):
            db.set_odoo_link(
                job_id,
                model=str(dossier["model"]),
                record_id=int(dossier["record_id"]),
                message_id=None,
            )
        # Enregistré maintenant, pas à la fin : ajouter « Acritec » doit
        # faire remonter les termes qui l'accompagnent dès la réunion
        # suivante, même si celle-ci échoue.
        db.record_vocabulary(
            [
                *(payload.context.get("glossary_terms") or []),
                *([payload.context["client_company"]] if payload.context.get("client_company") else []),
            ]
        )
        return {
            "job_id": job_id,
            "model": model,
            "estimated_cost_usd": estimate["cost_usd"],
            "chunks": [
                {"index": i, "start": start, "end": end}
                for i, (start, end) in enumerate(windows)
            ],
            "audio": AUDIO_PROFILE,
        }

    @app.put("/api/jobs/{job_id}/chunks/{index}", status_code=status.HTTP_202_ACCEPTED)
    async def upload_chunk(
        job_id: int, index: int, request: Request, owner_id: int = Depends(current_user)
    ) -> dict:
        job = owned_job(job_id, owner_id)
        # Une transcription interrompue ne reprend pas par la bande : le
        # navigateur qui encode encore s'arrête sur ce refus.
        if job["status"] == "cancelled" or job["deleted_at"]:
            raise HTTPException(status.HTTP_409_CONFLICT, "Transcription interrompue.")
        chunks = {c["idx"]: c for c in db.chunks_for_job(job_id)}
        window = chunks.get(index)
        if window is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Fenêtre inconnue.")

        body = await request.body()
        if not body:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Fenêtre audio vide.")
        if len(body) > MAX_CHUNK_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "Fenêtre audio trop lourde : le navigateur doit encoder en "
                "Opus mono 16 kHz.",
            )

        path = config.chunk_dir / f"job{job_id}_chunk{index}{CHUNK_SUFFIX}"
        path.write_bytes(body)
        db.set_chunk_status(job_id, index, "en_cours", error=None)
        db.set_job_status(job_id, "en_cours")

        # 202 tout de suite : Cloudflare coupe vers 100 s, la
        # transcription dure souvent plus longtemps.
        task = asyncio.create_task(_run_chunk(job, window, path))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return {"accepted": True, "index": index, "bytes": len(body)}

    @app.get("/api/jobs/{job_id}")
    def job_state(job_id: int, owner_id: int = Depends(current_user)) -> dict:
        job = owned_job(job_id, owner_id)
        chunks = db.chunks_for_job(job_id)
        return {
            "job_id": job_id,
            "status": job["status"],
            "error": job["error_message"],
            "filename": job["filename"],
            "model": job["model"],
            "title": job["title"],
            "cost_usd": job["cloud_cost_usd"],
            "chunks": [
                {
                    "index": c["idx"],
                    "start": c["start_second"],
                    "end": c["end_second"],
                    "status": c["status"],
                    "error": c["error"],
                }
                for c in chunks
            ],
            # Ce que le navigateur doit (ré)encoder : la reprise après
            # onglet fermé ou réseau coupé tient dans cette liste.
            "missing_chunks": [
                c["idx"] for c in chunks if c["status"] not in {"termine"}
            ],
        }

    def _finaliser(job_id: int) -> FinalizeResponse:
        """Fusionne les fenêtres et termine la réunion.

        Appelé par le serveur lui-même dès la dernière fenêtre
        transcrite : la personne peut avoir quitté l'écran, fermé
        l'onglet ou lancé une autre transcription entre-temps. Le
        navigateur n'est plus là que pour encoder et envoyer.
        """
        job = db.get_job(job_id) or {}
        chunks = db.chunks_for_job(job_id)
        results = [
            CloudChunkResult.from_dict(json.loads(c["result_json"] or "{}"))
            for c in chunks
        ]
        merged = merge_chunk_results(results)
        text = transcript_text(merged.segments)
        # Relancer une réunion pour rattraper une fenêtre ne doit pas
        # détruire la version déjà relue.
        db.archive_current_version(job_id)
        db.replace_segments(job_id, merged.segments)
        db.finish_job(
            job_id,
            title=merged.title,
            transcript=text,
            speakers=merged.speakers,
            technical_terms=merged.technical_terms,
            uncertain=merged.uncertain,
            cost_usd=merged.usage.cost_usd,
        )
        # Les noms que le modèle a reconnus s'appliquent aux répliques :
        # sans cela, la transcription affichait « sanaa.philippe@… » à côté
        # d'une carte qui disait déjà « Sanaa Philippe ».
        db.renommer_interlocuteurs(job_id, merged.speakers)
        for c in chunks:
            _discard(config.chunk_dir / f"job{job_id}_chunk{c['idx']}{CHUNK_SUFFIX}")
        text = (db.get_job(job_id) or {}).get("transcript") or text
        depot = _deposer_seul(db.get_job(job_id) or job, merged.title, text)
        db.noter_depot_odoo(job_id, depot)
        return FinalizeResponse(
            odoo=depot,
            uncertain=merged.uncertain,
            job_id=job_id,
            title=merged.title,
            transcript=text,
            speakers=merged.speakers,
            technical_terms=merged.technical_terms,
            cost_usd=merged.usage.cost_usd,
        )

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int, owner_id: int = Depends(current_user)) -> dict:
        """Interrompt une transcription lancée.

        Les fenêtres pas encore transcrites ne le seront pas, le navigateur
        qui encode encore s'arrête au prochain envoi, et la réunion part à
        la corbeille — récupérable, comme tout ce qu'on jette. Une
        transcription terminée ne s'interrompt plus : elle se jette.

        Un retraitement interrompu, lui, reste dans la bibliothèque : la
        réunion existait avant, avec sa vidéo, et peut se relancer.
        """
        job = owned_job(job_id, owner_id)
        if job["status"] in {"termine", "finalisation"}:
            raise HTTPException(status.HTTP_409_CONFLICT, "Transcription déjà terminée.")
        db.set_job_status(job_id, "cancelled", error="Interrompue à la demande.")
        if not json.loads(job["context_json"] or "{}").get("reprocessed"):
            db.jeter_job(job_id)
        return {"cancelled": True}

    @app.post("/api/client-errors", status_code=status.HTTP_204_NO_CONTENT,
              response_class=Response)
    def client_error(
        report: ClientError, request: Request, owner_id: int = Depends(current_user)
    ) -> Response:
        """Ce que le navigateur a vu échouer, dans les journaux du serveur.

        Une erreur d'encodage ou d'envoi n'arrive jamais jusqu'ici d'elle-
        même : sans ce relais, on ne sait d'un « network error » que ce
        qu'en dit la personne qui l'a eu."""
        log.warning(
            "erreur navigateur : utilisateur=%s job=%s étape=%s fichier=%s message=%s agent=%s",
            owner_id, report.job_id, report.stage, report.file_name[:120],
            report.message[:500], request.headers.get("user-agent", "")[:160],
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/jobs/{job_id}/finalize", response_model=FinalizeResponse)
    def finalize(job_id: int, owner_id: int = Depends(current_user)) -> FinalizeResponse:
        """Porte de secours : le serveur finalise seul, mais une réunion
        restée « à finaliser » (serveur redémarré au mauvais moment) se
        débloque ici. Déjà terminée, elle est simplement relue."""
        job = owned_job(job_id, owner_id)
        if job["status"] == "termine":
            return FinalizeResponse(
                job_id=job_id,
                title=job["title"] or "",
                transcript=job["transcript"] or "",
                speakers=json.loads(job["speaker_map_json"] or "{}"),
                technical_terms=json.loads(job["technical_terms_json"] or "[]"),
                uncertain=json.loads(job["uncertain_json"] or "[]"),
                cost_usd=float(job["cloud_cost_usd"] or 0),
                odoo=json.loads(job["odoo_depot_json"] or "{}"),
            )
        chunks = db.chunks_for_job(job_id)
        missing = [c["idx"] for c in chunks if c["status"] != "termine"]
        if missing:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "Fenêtres encore manquantes : " + ", ".join(str(i) for i in missing),
            )
        if not db.reserver_finalisation(job_id):
            raise HTTPException(status.HTTP_409_CONFLICT, "Finalisation déjà en cours.")
        return _finaliser(job_id)

    # -- bibliothèque --------------------------------------------------

    def _deposer_seul(job: dict, titre: str, transcript: str) -> dict:
        """Dépose la transcription et prévient, si tout était décidé.

        La transcription est déjà enregistrée quand on arrive ici : un
        dépôt raté ne doit donc rien casser, seulement se raconter. Le
        bouton « Déposer dans Odoo » reste la porte de secours.
        """
        modele = str(job["odoo_model"] or "")
        record_id = int(job["odoo_record_id"] or 0)
        contexte = json.loads(job["context_json"] or "{}")
        if not (modele and record_id) or job["odoo_message_id"]:
            return {}
        if not contexte.get("odoo_auto"):
            return {"pending": True, "model": modele, "record_id": record_id}

        owner_id = int(job["owner_id"])
        try:
            passerelle = odoo_pour(owner_id)
            chatter = passerelle.chatter()
            message_id = chatter.publier(
                modele,
                record_id,
                composer(titre or "Transcription complète", transcript,
                         entete="Déposé automatiquement par transcript.ekonum.fr."),
            )
        except (ChatterError, OdooUnavailable, CoffreIndisponible) as exc:
            log.warning("dépôt automatique impossible (job %s) : %s", job["id"], exc)
            return {"published": False, "error": str(exc),
                    "model": modele, "record_id": record_id}

        db.set_odoo_link(int(job["id"]), model=modele, record_id=record_id,
                         message_id=message_id)
        return {
            "published": True,
            "message_id": message_id,
            "model": modele,
            "record_id": record_id,
            **_prevenir(passerelle, chatter, modele, record_id, titre),
        }

    def _prevenir(passerelle, chatter, modele: str, record_id: int, titre: str) -> dict:
        """Dit que c'est fait, là où la personne le verra.

        La conversation OdooBot d'abord, une activité sur le dossier à
        défaut. Ne pas prévenir n'annule pas le dépôt : c'est signalé,
        pas fatal.
        """
        lien = f"{config.odoo_url.rstrip('/')}/odoo/{modele.split('.')[0]}/{record_id}"
        texte = (
            f'Transcription déposée : <b>{html.escape(titre or "réunion")}</b> — '
            f'<a href="{html.escape(lien)}">ouvrir le dossier</a>.'
        )
        try:
            identite = passerelle.identite()
            message = chatter.prevenir(identite["partner_id"], texte)
            if message:
                return {"notified": "odoobot"}
            activite = chatter.activite(
                modele, record_id, identite["user_id"],
                f"Transcription déposée : {titre or 'réunion'}",
                "Déposée automatiquement, à relire.",
            )
            return {"notified": "activite" if activite else "aucun"}
        except (ChatterError, OdooUnavailable) as exc:
            log.warning("ping impossible : %s", exc)
            return {"notified": "aucun", "notify_error": str(exc)}

    @app.delete("/api/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT,
                response_class=Response)
    def jeter(job_id: int, owner_id: int = Depends(current_user)) -> Response:
        """À la corbeille, pas au néant.

        Une transcription représente parfois une heure de réunion et
        quelques dizaines de centimes : la perdre sur un clic de travers
        serait une faute. Elle reste récupérable pendant le délai de
        rétention, puis disparaît pour de bon.
        """
        owned_job(job_id, owner_id)
        db.jeter_job(job_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.delete("/api/trash", status_code=status.HTTP_200_OK)
    def vider_corbeille(owner_id: int = Depends(current_user)) -> dict:
        """Supprime pour de bon tout ce qui est à la corbeille.

        Seulement la corbeille : une réunion active ne peut pas
        disparaître par ce chemin, il faut d'abord l'y avoir mise.
        """
        supprimees: list[int] = []
        for job in db.list_jobs(owner_id, limit=10_000, etat="corbeille"):
            try:
                effacer(int(job["id"]))
                supprimees.append(int(job["id"]))
            except StockageIndisponible as exc:
                log.warning("réunion %s gardée à la corbeille : %s", job["id"], exc)
        return {"deleted": supprimees}

    @app.delete("/api/jobs/{job_id}/permanent", status_code=status.HTTP_204_NO_CONTENT,
                response_class=Response)
    def supprimer_definitivement(
        job_id: int, owner_id: int = Depends(current_user)
    ) -> Response:
        """Efface une réunion déjà jetée, sans attendre la rétention."""
        job = owned_job(job_id, owner_id)
        if not job["deleted_at"]:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "Mets d'abord la réunion à la corbeille : on ne supprime pas "
                "définitivement ce qu'on n'a pas choisi de jeter.",
            )
        try:
            effacer(job_id)
        except StockageIndisponible as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # -- vidéo compressée ------------------------------------------------

    @app.post("/api/jobs/{job_id}/video")
    def ouvrir_envoi_video(
        job_id: int, payload: EnvoiVideo, owner_id: int = Depends(current_user)
    ) -> dict:
        """Prépare l'envoi de la vidéo compressée d'une réunion.

        La session Drive reste ici : le navigateur n'envoie que des
        morceaux à transcript, qui les relaie.
        """
        job = owned_job(job_id, owner_id)
        if job["video_file_id"] and not payload.remplacer:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "Cette réunion a déjà sa vidéo."
            )
        try:
            session = stockage().ouvrir_envoi(
                f"transcript-{job_id}.mp4", payload.taille, payload.type or "video/mp4"
            )
        except StockageIndisponible as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        db.ouvrir_envoi_video(job_id, session, payload.taille)
        return {"morceau": MORCEAU}

    @app.put("/api/jobs/{job_id}/video")
    async def envoyer_morceau_video(
        job_id: int, debut: int, request: Request, owner_id: int = Depends(current_user)
    ) -> dict:
        job = owned_job(job_id, owner_id)
        session = job["video_session"]
        if not session:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "Aucun envoi de vidéo en cours pour cette réunion."
            )
        octets = await request.body()
        if not octets or len(octets) > MORCEAU:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Morceau vide ou trop gros (au plus {MORCEAU // (1024 * 1024)} Mio).",
            )
        total = int(job["video_bytes"] or 0)
        try:
            fichier_id = await asyncio.to_thread(
                stockage().envoyer_morceau, session, debut, octets, total
            )
        except StockageIndisponible as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
        if fichier_id:
            remplacee = db.terminer_envoi_video(job_id, fichier_id)
            # L'ancienne n'est effacée qu'une fois la nouvelle entière. Si
            # le stockage refuse, la réunion est à jour quand même : on le
            # note pour la retrouver, sans faire échouer l'envoi.
            if remplacee:
                try:
                    await asyncio.to_thread(stockage().supprimer, remplacee)
                except StockageIndisponible as exc:
                    log.warning("vidéo %s remplacée mais gardée (réunion %s) : %s",
                                remplacee, job_id, exc)
        return {"recu": debut + len(octets), "total": total, "termine": bool(fichier_id)}

    @app.get("/api/jobs/{job_id}/video")
    def lire_video(
        job_id: int, request: Request, owner_id: int = Depends(current_user)
    ):
        """La vidéo, en flux et par plages — c'est ce qui permet d'avancer
        dans la réunion sans la télécharger en entier.

        Seule la première plage d'une lecture est comptée : un lecteur
        en demande des dizaines en avançant, ce serait compter des
        sauts, pas des lectures.
        """
        job = owned_job(job_id, owner_id)
        if not job["video_file_id"]:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Pas de vidéo pour cette réunion.")
        plage = request.headers.get("range", "")
        try:
            reponse = stockage().lire(str(job["video_file_id"]), plage)
        except StockageIndisponible as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
        # Le navigateur qui la retraite la lit aussi : ce n'est pas une
        # lecture, et il en ouvrirait une par fenêtre. L'aperçu de l'écran
        # de lancement, une balise <video>, ne peut que le dire dans l'URL.
        traitement = "processing" in (
            request.headers.get("x-video-purpose"), request.query_params.get("purpose"),
        )
        if plage in ("", "bytes=0-") and not traitement:
            db.compter_lecture_video(job_id)
        return StreamingResponse(
            reponse.flux, status_code=reponse.statut, headers=reponse.entetes,
            media_type=reponse.entetes.get("Content-Type", "video/mp4"),
        )

    @app.post("/api/jobs/{job_id}/archive")
    def archiver(job_id: int, owner_id: int = Depends(current_user)) -> dict:
        """Hors de la bibliothèque, mais intacte et toujours cherchable."""
        owned_job(job_id, owner_id)
        db.archiver_job(job_id)
        return {"job_id": job_id, "etat": "archive"}

    @app.post("/api/jobs/{job_id}/restore")
    def restaurer(job_id: int, owner_id: int = Depends(current_user)) -> dict:
        """Ressort d'archive comme de corbeille : un seul geste."""
        owned_job(job_id, owner_id)
        db.restaurer_job(job_id)
        return {"job_id": job_id, "etat": "actif"}

    @app.get("/api/jobs")
    def list_jobs(
        response: Response,
        etat: str = "actif",
        sort: str = "date",
        order: str = "desc",
        page: int = 1,
        per_page: int = 100,
        owner_id: int = Depends(current_user),
    ) -> list[dict]:
        """Une page de la bibliothèque. La réponse reste une liste — les
        appels existants n'ont rien à changer — et le total arrive dans
        l'en-tête `X-Total-Count`, pour paginer."""
        per_page = min(max(per_page, 1), 200)
        page = max(page, 1)
        # La purge se fait ici plutôt que par une tâche planifiée : le
        # serveur tient dans 256 Mo et n'a pas d'ordonnanceur, et une
        # corbeille qu'on consulte est une corbeille qu'on peut vider.
        if etat == "corbeille":
            for perime in db.corbeille_perimee(config.corbeille_jours):
                try:
                    effacer(perime)
                    log.info("corbeille : réunion %s purgée", perime)
                except StockageIndisponible as exc:
                    log.warning("corbeille : réunion %s gardée : %s", perime, exc)
        response.headers["X-Total-Count"] = str(db.count_jobs(owner_id, etat))
        return [
            {
                "job_id": job["id"],
                "filename": job["filename"],
                "title": job["title"],
                "status": job["status"],
                "model": job["model"],
                "duration_seconds": job["duration_seconds"],
                "cost_usd": job["cloud_cost_usd"],
                "created_at": job["created_at"],
                "has_versions": bool(job["previous_versions_json"]),
                "archived_at": job["archived_at"],
                "deleted_at": job["deleted_at"],
                "meeting_date": job["meeting_date"],
                # De quoi retraiter depuis la liste : la vidéo stockée est
                # relue par le navigateur, qui doit en connaître le poids.
                "video_bytes": job["video_bytes"] if job["video_file_id"] else None,
                "reprocessable": job["status"] in REPROCESSABLE and not job["transcript"],
                # Seulement pour ce qui tourne encore : la bibliothèque
                # montre une réunion en cours progresser pendant qu'on en
                # lance une autre.
                "progress": _avancement(job) if job["status"] not in ("termine",) else None,
            }
            for job in db.list_jobs(
                owner_id, limit=per_page, etat=etat, sort=sort,
                descending=order != "asc", offset=(page - 1) * per_page,
            )
        ]

    def _avancement(job: dict) -> dict:
        fenetres = db.chunks_for_job(int(job["id"]))
        return {
            "done": sum(1 for c in fenetres if c["status"] == "termine"),
            "total": len(fenetres),
        }

    @app.post("/api/jobs/import", status_code=status.HTTP_200_OK)
    def import_job(payload: ImportedJob, owner_id: int = Depends(current_user)) -> dict:
        """Reprend une réunion déjà transcrite, sans repasser par Gemini.

        C'est la bascule de la bibliothèque macOS : pousser depuis le
        poste plutôt que d'aller écrire dans le volume du conteneur.
        Idempotent, donc relançable — une reprise de plusieurs années ne
        réussit jamais du premier coup.
        """
        job_id, nouveau = db.import_job(
            owner_id=owner_id,
            payload={
                **payload.model_dump(exclude={"segments", "meeting_date"}),
                "segments": [s.model_dump() for s in payload.segments],
            },
        )
        # Une date de réunion complète une reprise déjà faite, sans rien
        # écraser : c'est ce qui permet de la rattraper en relançant
        # l'import, qui ne la transmettait pas au début.
        if payload.meeting_date and not (db.get_job(job_id) or {}).get("meeting_date"):
            db.set_meeting_date(job_id, _date_reunion(payload.meeting_date))
        return {"job_id": job_id, "imported": nouveau}

    @app.get("/api/jobs/{job_id}/detail")
    def job_detail(job_id: int, owner_id: int = Depends(current_user)) -> dict:
        job = owned_job(job_id, owner_id)
        segments = db.segments_for_job(job_id)
        return {
            "job_id": job_id,
            "filename": job["filename"],
            "title": job["title"],
            "status": job["status"],
            "reprocessable": job["status"] in REPROCESSABLE and not job["transcript"],
            "model": job["model"],
            "duration_seconds": job["duration_seconds"],
            "cost_usd": job["cloud_cost_usd"],
            "transcript": job["transcript"] or "",
            "speakers": json.loads(job["speaker_map_json"] or "{}"),
            "technical_terms": json.loads(job["technical_terms_json"] or "[]"),
            # Ce dont le modèle n'était pas sûr : c'est la liste de ce
            # qu'il faut réécouter, et elle ne sert à rien si elle reste
            # dans la base.
            "uncertain": json.loads(job["uncertain_json"] or "[]"),
            "meeting_date": job["meeting_date"],
            "created_at": job["created_at"],
            "video": {
                "presente": bool(job["video_file_id"]),
                "en_cours": bool(job["video_session"]),
                "octets": job["video_bytes"],
                "deposee_le": job["video_uploaded_at"],
                "lectures": job["video_lectures"] or 0,
            },
            "segments": segments,
            # Les répliques des versions restent au serveur : elles pèsent,
            # et l'écran n'a besoin que du texte et de savoir si on peut
            # revenir à la version.
            "previous_versions": [
                {
                    **{k: v for k, v in version.items() if k != "segments"},
                    "restaurable": db.repliques_de_version(version, segments) is not None,
                }
                for version in json.loads(job["previous_versions_json"] or "[]")
            ],
            "odoo": {
                "depot": json.loads(job["odoo_depot_json"] or "{}"),
                "model": job["odoo_model"],
                "record_id": job["odoo_record_id"],
                "message_id": job["odoo_message_id"],
                "published_at": job["odoo_published_at"],
            },
        }

    @app.patch("/api/jobs/{job_id}")
    def patch_context(
        job_id: int, patch: ContextPatch, owner_id: int = Depends(current_user)
    ) -> dict:
        owned_job(job_id, owner_id)
        db.update_job_context(
            job_id,
            title=patch.title,
            technical_terms=patch.technical_terms,
        )
        if patch.speakers is not None:
            db.renommer_interlocuteurs(job_id, patch.speakers)
        if patch.meeting_date is not None:
            db.set_meeting_date(job_id, _date_reunion(patch.meeting_date))
        return {"updated": True}

    @app.post("/api/jobs/{job_id}/terms/replace")
    def replace_term_route(
        job_id: int, payload: TermReplacement, owner_id: int = Depends(current_user)
    ) -> dict:
        job = owned_job(job_id, owner_id)
        transcript, segments, terms, occurrences = replace_term(
            transcript=job["transcript"] or "",
            segments=db.segments_for_job(job_id),
            technical_terms=json.loads(job["technical_terms_json"] or "[]"),
            old=payload.old,
            new=payload.new,
        )
        if occurrences:
            db.set_transcript(job_id, transcript)
            # `segments_for_job` renvoie des colonnes SQL ; `replace_segments`
            # attend le vocabulaire des segments cloud.
            db.replace_segments(
                job_id,
                [
                    {
                        "start": s["start_second"],
                        "end": s["end_second"],
                        "speaker": s["speaker"],
                        "text": s["text"],
                    }
                    for s in segments
                ],
            )
        db.update_job_context(job_id, technical_terms=terms)
        # Un mot corrigé n'est plus à vérifier : le laisser dans la liste
        # ferait relire ce qui est déjà réglé.
        passages = json.loads(job["uncertain_json"] or "[]")
        restants = [p for p in passages if not _designe(str(p.get("text") or ""), payload.old)]
        if len(restants) != len(passages):
            db.set_uncertain(job_id, restants)
        return {"occurrences": occurrences, "technical_terms": terms}

    @app.post("/api/jobs/{job_id}/versions/{index}/restore")
    def restaurer_version(
        job_id: int, index: int, owner_id: int = Depends(current_user)
    ) -> dict:
        """Revenir à une version antérieure — la version en place rejoint
        l'historique, donc le retour en arrière se défait lui aussi."""
        owned_job(job_id, owner_id)
        try:
            version = db.restaurer_version(job_id, index)
        except KeyError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Version inconnue.") from exc
        except ValueError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return {"title": version.get("title") or ""}

    @app.post("/api/jobs/{job_id}/chunks/{index}/reset", status_code=status.HTTP_200_OK)
    def reset_chunk(
        job_id: int, index: int, owner_id: int = Depends(current_user)
    ) -> dict:
        job = owned_job(job_id, owner_id)
        if index < 0 or index >= int(job["chunk_count"]):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Fenêtre inconnue.")
        db.reset_chunk(job_id, index)
        # Le job repart en attente : `missing_chunks` guidera le navigateur
        # vers cette seule fenêtre, sans repayer les autres.
        db.set_job_status(job_id, "en_attente")
        return {"reset": index}

    # -- enrôlement d'un appareil ----------------------------------------
    #
    # Le motif est celui des téléviseurs : l'appareil affiche un code
    # court, la personne l'ouvre dans son navigateur — déjà authentifiée
    # par Access — et valide. L'appareil repart avec son propre jeton.
    # Rien à recopier, aucun mot de passe nulle part.

    def _enrolement_ouvert() -> None:
        if not config.enrolement:
            # 404 plutôt que 403 : tant que la fonction dort, elle
            # n'existe pas.
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Chemin inconnu.")

    @app.post("/api/enroll/device", status_code=status.HTTP_201_CREATED)
    def enroler_appareil(payload: DemandeAppareil) -> dict:
        """Ouvre une demande. Seule route sans authentification.

        Elle ne donne rien : un code appareil inutile tant qu'un humain
        n'a pas validé dans son navigateur, et qui expire en quinze
        minutes.
        """
        _enrolement_ouvert()
        db.purger_enrolements()
        code_appareil, code_humain, echeance = db.ouvrir_enrolement(payload.appareil)
        return {
            "code_appareil": code_appareil,
            "code_humain": code_humain,
            # La racine, pas un chemin dédié : l'interface est une page
            # unique, et seul ce paramètre la fait bifurquer.
            "url": f"{config.public_url.rstrip('/')}/?code={code_humain}",
            "expire_le": echeance,
        }

    @app.get("/api/enroll/{code_humain}")
    def voir_enrolement(code_humain: str, owner_id: int = Depends(human_user)) -> dict:
        """Ce que la personne doit voir avant de valider : quel appareil,
        demandé quand."""
        _enrolement_ouvert()
        demande = db.enrolement_par_code_humain(code_humain)
        if not demande:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Code inconnu ou expiré.")
        return {
            "appareil": demande["appareil"],
            "statut": demande["statut"],
            "demande_le": demande["created_at"],
            "expire_le": demande["expires_at"],
        }

    @app.post("/api/enroll/{code_humain}/approve")
    def approuver_enrolement(
        code_humain: str, owner_id: int = Depends(human_user)
    ) -> dict:
        """Valide l'appareil, sous **son** identité.

        `human_user` : un jeton d'API ne peut pas enrôler un appareil de
        plus. Sinon un jeton volé se multiplierait tout seul.
        """
        _enrolement_ouvert()
        if not db.approuver_enrolement(code_humain, owner_id):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "Demande inconnue, expirée ou déjà traitée. Relance "
                "l'enrôlement depuis l'application.",
            )
        return {"statut": "approuve"}

    @app.post("/api/enroll/token")
    def reclamer_jeton(payload: CodeAppareil) -> dict:
        """L'appareil vient chercher son jeton, une seule fois."""
        _enrolement_ouvert()
        vue = db.reclamer_enrolement(payload.code_appareil)
        if vue["statut"] == "inconnu":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Demande inconnue.")
        return vue

    @app.get("/api/me")
    def me(request: Request, owner_id: int = Depends(current_user)) -> dict:
        """Qui suis-je, et comment suis-je entré.

        L'interface l'affiche : savoir sous quel compte on travaille est
        la première chose qu'on cherche sur un outil d'équipe, et son
        absence rendait la page anonyme.
        """
        par_jeton = request.headers.get("Authorization", "").startswith("Bearer ")
        return {
            "email": db.email_for_user(owner_id),
            "via": "jeton d'API" if par_jeton else "Cloudflare Access",
        }

    @app.get("/api/me/odoo")
    def voir_odoo(owner_id: int = Depends(current_user)) -> dict:
        """État du raccordement Odoo. **Ne rend jamais la clé.**"""
        login, chiffree = db.odoo_credentials(owner_id)
        return {
            "configured": bool(login and chiffree),
            "login": login,
            "server": config.odoo_url,
            "chiffrement_disponible": coffre.disponible,
        }

    @app.put("/api/me/odoo")
    def poser_odoo(
        payload: OdooCredentials, owner_id: int = Depends(human_user)
    ) -> dict:
        """Enregistre la clé API personnelle, chiffrée.

        Réservé à une connexion humaine : un jeton d'API ne doit pas
        pouvoir déposer une identité Odoo à la place de quelqu'un.
        """
        try:
            chiffree = coffre.chiffrer(payload.api_key)
        except CoffreIndisponible as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)
            ) from exc
        db.set_odoo_credentials(owner_id, login=payload.login, key_chiffree=chiffree)
        return {"configured": True, "login": payload.login.strip()}

    @app.delete("/api/me/odoo", status_code=status.HTTP_204_NO_CONTENT,
                response_class=Response)
    def retirer_odoo(owner_id: int = Depends(human_user)) -> Response:
        db.clear_odoo_credentials(owner_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/tokens", status_code=status.HTTP_201_CREATED)
    def create_token(payload: TokenRequest, owner_id: int = Depends(human_user)) -> dict:
        """Fabrique un jeton. **La valeur n'est renvoyée qu'ici.**"""
        token_id, secret = db.create_api_token(owner_id, payload.name)
        return {
            "id": token_id,
            "name": payload.name.strip() or "sans nom",
            "token": secret,
            "avertissement": "Ce jeton ne sera plus jamais affiché.",
        }

    @app.get("/api/tokens")
    def list_tokens(owner_id: int = Depends(human_user)) -> list[dict]:
        return db.list_api_tokens(owner_id)

    @app.delete("/api/tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT,
                response_class=Response)
    def revoke_token(token_id: int, owner_id: int = Depends(human_user)) -> Response:
        if not db.revoke_api_token(owner_id, token_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Jeton introuvable.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/vocabulary")
    def vocabulary(selected: str = "", _: int = Depends(current_user)) -> list[dict]:
        """Suggestions de vocabulaire, communes à l'équipe."""
        return db.suggest_vocabulary(
            [t.strip() for t in selected.split(",") if t.strip()]
        )

    @app.post("/api/vocabulary")
    def record_vocabulary(
        payload: VocabularyRecord, _: int = Depends(current_user)
    ) -> dict:
        db.record_vocabulary(payload.terms)
        return {"recorded": len(payload.terms)}

    @app.delete(
        "/api/vocabulary/{term}",
        status_code=status.HTTP_204_NO_CONTENT,
        # Sans cette classe, FastAPI prépare une réponse JSON et refuse le
        # 204, qui n'a par définition pas de corps.
        response_class=Response,
    )
    def forget_vocabulary(term: str, _: int = Depends(current_user)) -> Response:
        db.forget_vocabulary(term)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/odoo/records")
    def odoo_records(q: str = "", owner_id: int = Depends(current_user)) -> dict:
        """Dossiers candidats, pour choisir où déposer."""
        passerelle = odoo_pour(owner_id)
        if not passerelle.configured:
            return {
                "available": False,
                "reason": "Ajoute ta clé API Odoo dans ton compte.",
                "records": [],
            }
        try:
            return {"available": True, "records": passerelle.search_records(q)}
        except OdooUnavailable as exc:
            log.warning("recherche Odoo indisponible : %s", exc)
            return {"available": False, "reason": str(exc), "records": []}

    @app.post("/api/jobs/{job_id}/enrich")
    def reenrichir(
        job_id: int,
        payload: Reenrichissement,
        owner_id: int = Depends(current_user),
    ) -> dict:
        """Refait titre, noms et corrections sans retranscrire.

        C'est ce qui rend une mauvaise liaison peu coûteuse : quand le
        dossier retenu était le mauvais, il suffit d'en désigner un
        autre et de relire le texte à sa lumière. Une transcription
        complète coûte cent fois plus.
        """
        job = owned_job(job_id, owner_id)
        transcript = (job["transcript"] or "").strip()
        if not transcript:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "Cette réunion n'a pas de transcription."
            )

        contexte = json.loads(job["context_json"] or "{}")
        if payload.model and payload.record_id:
            # Un dossier désigné remplace celui d'avant, contexte compris.
            passerelle = odoo_pour(owner_id)
            if not passerelle.configured:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "Ajoute ta clé API Odoo dans ton compte pour charger ce dossier.",
                )
            try:
                pack = passerelle.context_pack(payload.model, payload.record_id)
            except OdooUnavailable as exc:
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
            contexte = {
                **contexte,
                "client_company": pack.get("client_company") or "",
                "glossary_terms": pack.get("terms") or [],
                "odoo_context": pack.get("summary") or "",
            }
            db.set_odoo_link(
                job_id, model=payload.model, record_id=payload.record_id,
                message_id=job["odoo_message_id"],
            )
            db.update_job_context_json(job_id, contexte)

        try:
            enrichi, usage = enrich_transcript_via_gemini(
                keys.get(),
                MODELE_ENRICHISSEMENT,
                transcript,
                language=str(job["language"] or "fr"),
                glossary_terms=list(contexte.get("glossary_terms") or []),
                expected_speaker_names=list(contexte.get("expected_speaker_names") or []),
                meeting_context=str(contexte.get("meeting_context") or ""),
                odoo_context=str(contexte.get("odoo_context") or ""),
            )
        except (CloudTranscriptionError, SecretError) as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

        db.add_api_usage(
            job_id=job_id,
            provider=provider_for_model(MODELE_ENRICHISSEMENT),
            model=usage.model or MODELE_ENRICHISSEMENT,
            step="reenrichissement",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )
        nouveau_titre = enrichi.get("title") or job["title"]
        noms = {k: v for k, v in (enrichi.get("speakers") or {}).items() if v and v != k}
        change = nouveau_titre != job["title"] or bool(noms) or (
            (enrichi.get("technical_terms") or None) is not None
            and enrichi.get("technical_terms") != json.loads(job["technical_terms_json"] or "[]")
        )
        # On n'empile une version que si la relecture change quelque
        # chose : sinon chaque dépôt manuel, qui relit d'abord, ajoutait
        # une « version antérieure » identique à l'actuelle.
        if change:
            db.archive_current_version(job_id)
        db.update_job_context(
            job_id,
            title=nouveau_titre,
            technical_terms=enrichi.get("technical_terms") or None,
        )
        if noms:
            db.renommer_interlocuteurs(job_id, {
                **json.loads(job["speaker_map_json"] or "{}"), **noms,
            })
        db.set_uncertain(job_id, enrichi.get("uncertain_passages") or [])
        return {
            "job_id": job_id,
            "title": enrichi.get("title") or job["title"],
            "speakers": enrichi.get("speakers") or {},
            "technical_terms": enrichi.get("technical_terms") or [],
            "corrections": enrichi.get("corrections") or [],
            "uncertain": enrichi.get("uncertain_passages") or [],
            "cost_usd": usage.cost_usd,
        }

    @app.post("/api/jobs/{job_id}/odoo/publish")
    def publier_odoo(
        job_id: int, payload: OdooPublication, owner_id: int = Depends(current_user)
    ) -> dict:
        """Dépose la transcription dans le chatter, en accordéon.

        Remplace la recopie manuelle qui a fini par gonfler des
        opportunités jusqu'à 300 000 caractères. L'accordéon garde le
        texte intégral disponible sans noyer l'historique commercial.
        """
        job = owned_job(job_id, owner_id)
        transcript = (job["transcript"] or "").strip()
        if not transcript:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "Cette réunion n'a pas de transcription à déposer.",
            )
        if job["odoo_message_id"]:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "Déjà déposée dans Odoo — la reposter empilerait deux copies.",
            )

        corps = composer(
            job["title"] or "Transcription complète",
            transcript,
            entete=payload.header,
        )
        try:
            passerelle = odoo_pour(owner_id)
        except CoffreIndisponible as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        if not passerelle.configured:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "Aucune clé API Odoo personnelle enregistrée. La note doit "
                "porter ton identité, pas celle d'un compte partagé — "
                "ajoute ta clé dans ton compte.",
            )
        try:
            message_id = passerelle.chatter().publier(
                payload.model, payload.record_id, corps
            )
        except (ChatterError, OdooUnavailable) as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

        db.set_odoo_link(
            job_id,
            model=payload.model,
            record_id=payload.record_id,
            message_id=message_id,
        )
        return {"message_id": message_id, "model": payload.model,
                "record_id": payload.record_id}

    @app.post("/api/probe")
    async def sonder(
        request: Request, moment: str = "", owner_id: int = Depends(current_user)
    ) -> dict:
        """Écoute le début d'un enregistrement et propose un dossier.

        Une fenêtre courte suffit pour savoir de qui et de quoi on
        parle ; le dossier retenu fournira ensuite le contexte de la
        vraie transcription. La sonde n'écrit rien dans Odoo : elle
        propose, c'est tout.
        """
        corps = await request.body()
        if not corps:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Fenêtre audio vide.")
        if len(corps) > MAX_SONDE_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "Fenêtre trop lourde pour une sonde : cinq minutes suffisent.",
            )

        chemin = config.chunk_dir / f"sonde{owner_id}_{int(time.time()*1000)}{CHUNK_SUFFIX}"
        chemin.write_bytes(corps)
        try:
            indices = await asyncio.to_thread(
                identifier, str(chemin), api_key=keys.get()
            )
        except (CloudTranscriptionError, SecretError) as exc:
            log.warning("sonde en échec : %s", exc)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
        finally:
            _discard(chemin)

        db.add_api_usage(
            job_id=None,
            provider=fournisseur(),
            model=indices.usage.model,
            step="sonde_identification",
            input_tokens=indices.usage.input_tokens,
            output_tokens=indices.usage.output_tokens,
            cost_usd=indices.usage.cost_usd,
        )

        # L'enquête coûte du jugement, pas de l'audio : on la mène dans
        # un fil pour ne pas bloquer la boucle d'événements pendant ses
        # allers-retours avec Odoo.
        enquete = await asyncio.to_thread(_enqueter, owner_id, indices, moment)
        db.add_api_usage(
            job_id=None,
            provider=fournisseur(MODELE_ENQUETE),
            model=enquete.usage.model or MODELE_ENQUETE,
            step="sonde_enquete",
            input_tokens=enquete.usage.input_tokens,
            output_tokens=enquete.usage.output_tokens,
            cost_usd=enquete.usage.cost_usd,
        )

        return {
            "clues": indices.to_dict(),
            # Guider la recherche n'a de sens qu'avec une clé Odoo : sans
            # elle, l'interface ne le propose pas.
            "odoo": odoo_configure(owner_id),
            "cost_usd": round(indices.usage.cost_usd + enquete.usage.cost_usd, 6),
            "investigation": {
                **enquete.to_dict(),
                # L'interface n'a pas à connaître le seuil : elle a
                # besoin de savoir si elle doit demander.
                "auto": bool(
                    enquete.dossier
                    and liaison_sans_demander(enquete.confiance, config.liaison_auto)
                ),
            },
            # Le repli : la recherche directe reste là quand l'enquête
            # renonce, pour que la personne ait quand même une liste.
            # Le retenu d'abord, ses alternatives ensuite : se corriger
            # doit coûter un clic, pas une nouvelle enquête.
            "candidates": (
                [enquete.dossier, *enquete.autres] if enquete.dossier else _candidats(
                    owner_id, indices.groupes_de_recherche(config.sonde_ignore)
                )
            ),
        }

    @app.post("/api/odoo/investigate")
    async def guided_investigation(
        demande: GuidedInvestigation, owner_id: int = Depends(current_user)
    ) -> dict:
        """Une nouvelle enquête, guidée par ce que dit la personne.

        Pas de seconde enquête de confirmation ici : rien ne se lie sans
        un clic, puisque c'est la personne qui choisit parmi ce qui
        revient.
        """
        if demande.messages[-1].role != "user":
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Rien à quoi répondre.")
        rejected = [r.model_dump() for r in demande.rejected]
        enquete = await asyncio.to_thread(
            _enqueter,
            owner_id,
            demande.clues,
            demande.moment,
            echanges=[m.model_dump() for m in demande.messages],
            ecartes=rejected,
        )
        db.add_api_usage(
            job_id=None,
            provider=fournisseur(MODELE_ENQUETE),
            model=enquete.usage.model or MODELE_ENQUETE,
            step="sonde_enquete_guidee",
            input_tokens=enquete.usage.input_tokens,
            output_tokens=enquete.usage.output_tokens,
            cost_usd=enquete.usage.cost_usd,
        )
        refuses = {(r["model"], r["id"]) for r in rejected}
        vus: set[tuple[str, int]] = set()
        propositions = []
        # Le modèle a pu reproposer un refusé malgré la consigne : on ne
        # le montre pas une seconde fois.
        for dossier in ([enquete.dossier] if enquete.dossier else []) + enquete.autres:
            cle = (dossier.get("model"), dossier.get("id"))
            if cle in refuses or cle in vus:
                continue
            vus.add(cle)
            propositions.append(dossier)
        return {
            "answer": enquete.raison or (
                "Rien trouvé qui colle." if not propositions else ""
            ),
            "candidates": propositions,
            "trace": enquete.journal,
            "cost_usd": round(enquete.usage.cost_usd, 6),
        }

    def odoo_configure(owner_id: int) -> bool:
        try:
            return bool(odoo_pour(owner_id).configured)
        except CoffreIndisponible:
            return False

    def _enqueter(
        owner_id: int,
        indices,
        moment: str = "",
        *,
        echanges: list[dict] | None = None,
        ecartes: list[dict] | None = None,
    ) -> Conclusion:
        """Mène l'enquête avec la clé Odoo de la personne, ou renonce.

        Odoo enrichit, il ne conditionne pas : sans clé, pas d'enquête,
        et la transcription reste possible.
        """
        try:
            passerelle = odoo_pour(owner_id)
        except CoffreIndisponible:
            return Conclusion(raison="Coffre indisponible.")
        if not passerelle.configured:
            return Conclusion(raison="Aucune clé API Odoo personnelle.")
        guidee = bool(echanges)
        try:
            return (enqueter if guidee else enqueter_confirme)(
                indices if isinstance(indices, dict) else indices.to_dict(),
                chercher=lambda terme, modeles: passerelle.search_records(
                    terme, modeles=modeles
                ),
                lire=passerelle.resume_dossier,
                # L'agenda du moment de l'enregistrement : une réunion
                # inscrite à cette heure-là pointe souvent déjà le
                # dossier, et c'est le signal le plus fiable.
                agenda=lambda: passerelle.meetings(
                    near=_instant(moment), window_hours=3.0
                ),
                api_key=keys.get(),
                **({"echanges": echanges, "ecartes": ecartes} if guidee else {}),
            )
        except (CloudTranscriptionError, SecretError) as exc:
            log.warning("enquête impossible : %s", exc)
            return Conclusion(raison=f"Enquête impossible : {exc}")

    def _candidats(
        owner_id: int, groupes: list[list[str]], limite: int = 5
    ) -> list[dict]:
        """Les dossiers qui collent aux indices, sans jamais bloquer.

        On s'arrête au premier groupe qui trouve : une fois la société
        reconnue, chercher aussi les personnes et les sujets ne fait que
        noyer le bon dossier.

        Odoo enrichit, il ne conditionne pas : pas de clé, pas de
        réseau, pas de candidats — et la transcription reste possible.
        """
        try:
            passerelle = odoo_pour(owner_id)
        except CoffreIndisponible:
            return []
        if not passerelle.configured:
            return []
        trouves: list[dict] = []
        vus: set[tuple[str, int]] = set()
        for groupe in groupes:
            for terme in groupe[:4]:
                try:
                    lignes = passerelle.search_records(terme, limit=limite)
                except OdooUnavailable as exc:
                    log.warning("recherche Odoo indisponible : %s", exc)
                    return trouves
                for ligne in lignes:
                    cle = (ligne["model"], int(ligne["id"]))
                    if cle in vus:
                        continue
                    vus.add(cle)
                    # Le terme qui a trouvé le dossier vaut explication :
                    # « proposé parce qu'on a entendu Acritec ».
                    trouves.append({**ligne, "matched": terme})
                    if len(trouves) >= limite:
                        return trouves
            if trouves:
                return trouves
        return trouves

    @app.get("/api/odoo/meetings")
    def odoo_meetings(owner_id: int = Depends(current_user)) -> dict:
        """Réunions du moment, pour proposer « c'est celle-là ».

        Une panne Odoo n'est pas une erreur ici : elle rend simplement la
        liste vide et le dit. Odoo enrichit, il ne conditionne pas — une
        réunion doit se transcrire même si le serveur est en maintenance.
        """
        passerelle = odoo_pour(owner_id)
        if not passerelle.configured:
            return {
                "available": False,
                "reason": "Ajoute ta clé API Odoo dans ton compte.",
                "meetings": [],
            }
        try:
            return {"available": True, "meetings": passerelle.meetings()}
        except OdooUnavailable as exc:
            log.warning("Odoo indisponible : %s", exc)
            return {"available": False, "reason": str(exc), "meetings": []}

    @app.get("/api/odoo/context")
    def odoo_context(
        model: str, record_id: int, owner_id: int = Depends(current_user)
    ) -> dict:
        """Pack de contexte d'une réunion : résumé, termes, société cliente."""
        passerelle = odoo_pour(owner_id)
        if not passerelle.configured:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Ajoute ta clé API Odoo dans ton compte.",
            )
        try:
            return passerelle.context_pack(model, record_id)
        except OdooUnavailable as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    google_secret = GeminiKey(  # lecteur de coffre générique, malgré son nom
        url=config.broker_url,
        token=config.broker_token,
        item=config.google_secret_item,
        field=config.google_secret_field,
    )
    register_recovery_routes(
        app,
        db=db,
        config=config,
        coffre=coffre,
        current_user=current_user,
        human_user=human_user,
        storage=stockage,
        google_secret=lambda: google_secret.get(),
        oauth_opener=google_opener,
    )

    @app.get("/api/onboarding")
    def onboarding(owner_id: int = Depends(current_user)) -> dict:
        """L'accueil s'affiche tant que l'essentiel n'est pas fait — la clé
        Odoo et une première réunion — et qu'on ne l'a pas masqué. Le Mac
        est facultatif : tout le monde n'utilisait pas l'app."""
        etat = db.onboarding_state(owner_id)
        return {
            "show": not etat["dismissed"] and not (etat["odoo"] and etat["transcription"]),
            "steps": {k: etat[k] for k in ("odoo", "mac", "transcription")},
        }

    @app.post("/api/onboarding/dismiss", status_code=status.HTTP_204_NO_CONTENT,
              response_class=Response)
    def onboarding_dismiss(owner_id: int = Depends(human_user)) -> Response:
        db.dismiss_onboarding(owner_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/settings")
    def settings_view(owner_id: int = Depends(current_user)) -> dict:
        """Ce que l'interface a besoin de savoir : les modèles offerts et
        où en est le budget d'équipe — la clé Gemini étant partagée, le
        plafond l'est aussi."""
        spent = db.month_spend_usd()
        return {
            "models": [
                {
                    "id": entry["id"],
                    "label": entry.get("label") or entry["id"],
                    "default": bool(entry.get("default")),
                }
                for entry in CLOUD_TRANSCRIPTION_MODELS
                if provider_for_model(entry["id"]) == "gemini"
            ],
            "corbeille": {"retention_jours": config.corbeille_jours},
            "video": {"disponible": stockage_disponible()},
            "budget": {
                "spent_usd": round(spent, 4),
                "cap_usd": config.monthly_budget_usd,
            },
            "odoo": {"configured": odoo_pour(owner_id).configured},
            # Le navigateur encode la fenêtre de la sonde avant d'avoir
            # créé un traitement : il lui faut le profil ici aussi.
            "audio": AUDIO_PROFILE,
            "probe": {"window_seconds": FENETRE_SECONDES},
        }

    @app.get("/api/search")
    def search(q: str = "", owner_id: int = Depends(current_user)) -> list[dict]:
        return db.search_segments(owner_id, q)

    # -- tâche de fond -------------------------------------------------

    def interrompue(job_id: int) -> bool:
        courant = db.get_job(job_id) or {}
        return courant.get("status") == "cancelled" or bool(courant.get("deleted_at"))

    async def _run_chunk(job: dict, window: dict, path: Path) -> None:
        job_id = int(job["id"])
        index = int(window["idx"])
        async with app.state.semaphore:
            # Interrompue pendant que la fenêtre attendait son tour : on ne
            # paie pas pour une transcription dont personne ne veut plus.
            if interrompue(job_id):
                _discard(path)
                return
            try:
                result = await asyncio.to_thread(_transcribe_blocking, job, window, path)
            except (CloudTranscriptionError, SecretError) as exc:
                log.warning("job %s fenêtre %s en échec : %s", job_id, index, exc)
                db.set_chunk_status(job_id, index, "erreur", error=str(exc))
                db.set_job_status(job_id, "erreur", error=str(exc))
                return
            except Exception as exc:  # garde-fou : une tâche de fond ne doit rien avaler
                log.exception("job %s fenêtre %s : erreur inattendue", job_id, index)
                db.set_chunk_status(job_id, index, "erreur", error=str(exc))
                db.set_job_status(job_id, "erreur", error=str(exc))
                return
            finally:
                _discard(path)

            db.set_chunk_status(job_id, index, "termine", error=None, result=result.to_dict())
            db.add_api_usage(
                job_id=job_id,
                provider=provider_for_model(str(job["model"])),
                model=str(job["model"]),
                step="cloud_transcription",
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cost_usd=result.usage.cost_usd,
            )
            if interrompue(job_id):
                return
            remaining = [
                c for c in db.chunks_for_job(job_id) if c["status"] != "termine"
            ]
            # Deux fenêtres peuvent finir au même instant : une seule
            # obtient la réservation, et donc la fusion.
            if not remaining and db.reserver_finalisation(job_id):
                try:
                    await asyncio.to_thread(_finaliser, job_id)
                except Exception:  # garde-fou : la tâche de fond ne doit rien avaler
                    log.exception("job %s : finalisation automatique en échec", job_id)
                    db.set_job_status(job_id, "a_finaliser")

    def _transcribe_blocking(job: dict, window: dict, path: Path) -> CloudChunkResult:
        context = context_for_chunk(
            chunk_index=int(window["idx"]),
            chunk_count=int(job["chunk_count"]),
            start=float(window["start_second"]),
            end=float(window["end_second"]),
            language=str(job["language"]),
            context=json.loads(job["context_json"] or "{}"),
        )
        return transcribe_chunk(
            str(path),
            model_id=str(job["model"]),
            context=context,
            api_key=keys.get(),
        )

    # Monté en dernier pour ne pas masquer les routes ci-dessus. Le client
    # est servi par le même conteneur : une seule application Cloudflare
    # Access protège l'ensemble.
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir), name="static")
    return app


def _date_reunion(texte: str) -> str | None:
    """Normalise une date de réunion saisie, ou la refuse franchement."""
    lu = _instant(texte)
    if texte.strip() and lu is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Date illisible : {texte!r}.")
    return lu.isoformat(timespec="minutes") if lu else None


def _instant(moment: str) -> datetime | None:
    """L'heure de l'enregistrement, telle que le navigateur la connaît.

    Le fichier porte sa date de dernière modification ; à défaut,
    « maintenant » reste une approximation utile pour un dépôt fait dans
    la foulée.
    """
    texte = (moment or "").strip()
    if not texte:
        return None
    try:
        lu = datetime.fromisoformat(texte.replace("Z", "+00:00"))
    except ValueError:
        log.info("moment illisible : %r", moment)
        return None
    return lu if lu.tzinfo else lu.replace(tzinfo=timezone.utc)


def _discard(path: Path) -> None:
    """Le disque du VPS est tendu : une fenêtre ne survit pas à son usage."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("fenêtre %s non supprimée : %s", path, exc)


# uvicorn monte l'application par la fabrique — « uvicorn app.main:create_app
# --factory » — pour qu'importer ce module (dans les tests, par exemple) ne
# déclenche ni lecture d'environnement ni ouverture de base.
