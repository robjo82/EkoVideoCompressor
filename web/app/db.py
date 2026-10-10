"""Schéma et accès SQLite.

Migrations à la main, façon sync-hub : ``CREATE TABLE IF NOT EXISTS``
plus un ``_ensure_column`` idempotent. Pas d'ORM — les requêtes sont
peu nombreuses et le VPS est à 256 Mo.

Écarts assumés par rapport à la base de l'app macOS : ``jobs`` gagne un
``owner_id``, ``source_path`` devient un simple nom de fichier (le
serveur ne voit jamais le média), et ``speaker_profiles`` disparaît avec
pyannote.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import unicodedata
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    email      TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id             INTEGER NOT NULL REFERENCES users(id),
    filename             TEXT NOT NULL,
    duration_seconds     REAL NOT NULL DEFAULT 0,
    model                TEXT NOT NULL,
    language             TEXT NOT NULL DEFAULT 'fr',
    status               TEXT NOT NULL DEFAULT 'en_attente',
    error_message        TEXT,
    chunk_count          INTEGER NOT NULL DEFAULT 1,
    context_json         TEXT,
    title                TEXT,
    transcript           TEXT,
    speaker_map_json     TEXT,
    technical_terms_json TEXT,
    cloud_cost_usd       REAL NOT NULL DEFAULT 0,
    compressed_bytes     INTEGER,
    previous_versions_json TEXT,
    created_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner_id, id DESC);

CREATE TABLE IF NOT EXISTS job_chunks (
    job_id       INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    start_second REAL NOT NULL,
    end_second   REAL NOT NULL,
    status       TEXT NOT NULL DEFAULT 'attendu',
    error        TEXT,
    result_json  TEXT,
    updated_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, idx)
);

CREATE TABLE IF NOT EXISTS transcription_segments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    start_second REAL NOT NULL,
    end_second   REAL NOT NULL,
    speaker      TEXT,
    text         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_job ON transcription_segments(job_id, start_second);

CREATE TABLE IF NOT EXISTS api_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    step          TEXT NOT NULL DEFAULT '',
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_usage_created ON api_usage(created_at);

-- Jetons d'API, pour les appels machine : un script, une intégration
-- Odoo, un serveur MCP. Seule l'empreinte est stockée — un vol de base
-- ne doit pas rendre les jetons utilisables, et personne (pas même
-- l'interface) ne peut réafficher un jeton après sa création.
-- Enrôlement d'un appareil : l'app macOS obtient un jeton sans qu'on
-- ait à recopier quoi que ce soit à la main. Le code appareil est
-- stocké haché, comme un jeton ; le code humain, court et lisible, ne
-- vaut que le temps de la validation.
CREATE TABLE IF NOT EXISTS enrolements (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    appareil          TEXT NOT NULL DEFAULT '',
    code_appareil_sha TEXT NOT NULL UNIQUE,
    code_humain       TEXT NOT NULL UNIQUE,
    owner_id          INTEGER REFERENCES users(id),
    statut            TEXT NOT NULL DEFAULT 'en_attente',
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_tokens (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id     INTEGER NOT NULL REFERENCES users(id),
    name         TEXT NOT NULL,
    token_sha256 TEXT NOT NULL UNIQUE,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at TEXT,
    revoked_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_tokens_owner ON api_tokens(owner_id, id DESC);

-- Vocabulaire métier, **partagé par toute l'équipe** : « Odoo »,
-- « Ekonum » et les noms de clients sont communs. C'est un gain net sur
-- l'app macOS, où le glossaire était cloisonné par machine.
CREATE TABLE IF NOT EXISTS vocabulary (
    term       TEXT PRIMARY KEY,
    usages     INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used  TEXT
);

-- Co-occurrences, stockées dans les deux sens comme le fait l'app macOS.
-- C'est ce qui fait remonter « CVR Contrôle » dès qu'on saisit
-- « Acritec », alors que CVR est globalement plus rare que Odoo ou
-- Ekonum : la suggestion place la co-occurrence avant l'usage brut.
CREATE TABLE IF NOT EXISTS vocabulary_pairs (
    term  TEXT NOT NULL,
    peer  TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (term, peer)
);

-- Recherche plein texte. Table FTS5 autonome plutôt qu'en contenu
-- externe : les segments sont réécrits en bloc à chaque finalisation,
-- et des déclencheurs sur un DELETE massif coûteraient plus qu'ils ne
-- rapportent. Les colonnes UNINDEXED évitent qu'un identifiant de job
-- ressorte comme un résultat de recherche.
-- Connexion Google de chacun, pour retrouver ses anciens enregistrements
-- dans son Drive. Le jeton de rafraîchissement est chiffré comme les clés
-- Odoo : il agit au nom de son propriétaire.
CREATE TABLE IF NOT EXISTS google_accounts (
    user_id           INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    email             TEXT NOT NULL,
    refresh_token_enc TEXT NOT NULL,
    scope             TEXT NOT NULL DEFAULT '',
    connected_at      TEXT NOT NULL
);

-- Une connexion OAuth en cours : l'état lie le retour de Google à la
-- personne qui l'a demandé, le vérificateur PKCE à cette demande-là.
CREATE TABLE IF NOT EXISTS oauth_states (
    state      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    verifier   TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- D'où vient chaque enregistrement récupéré. Commune à toute l'équipe :
-- un fichier du Drive partagé récupéré par l'un ne se récupère pas une
-- seconde fois par l'autre.
CREATE TABLE IF NOT EXISTS recovered_sources (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source              TEXT NOT NULL,
    external_id         TEXT NOT NULL,
    checksum            TEXT,
    size_bytes          INTEGER,
    name                TEXT,
    job_id              INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    user_id             INTEGER NOT NULL REFERENCES users(id),
    recovered_at        TEXT NOT NULL,
    original_trashed_at TEXT,
    UNIQUE (source, external_id)
);
CREATE INDEX IF NOT EXISTS idx_recovered_checksum ON recovered_sources (checksum);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text,
    job_id       UNINDEXED,
    start_second UNINDEXED,
    speaker      UNINDEXED
);
"""


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn) -> None:
        """Migrations additives, jouées à chaque démarrage.

        La base de production porte déjà l'historique repris : les
        nouvelles colonnes s'ajoutent, la table ne se recrée pas.
        """
        for colonne, ddl in (
            ("odoo_login", "TEXT"),
            ("odoo_key_chiffree", "TEXT"),
            # L'invitation à récupérer son historique ne revient pas une
            # fois écartée.
            ("recovery_dismissed_at", "TEXT"),
            # L'accueil des nouveaux venus, une fois masqué, ne revient pas.
            ("onboarding_dismissed_at", "TEXT"),
        ):
            self._ensure_column(conn, "users", colonne, ddl)

        for colonne, ddl in (
            ("odoo_model", "TEXT"),
            ("odoo_record_id", "INTEGER"),
            ("odoo_message_id", "INTEGER"),
            ("odoo_published_at", "TEXT"),
            # Corbeille et archives : une réunion ne disparaît pas d'un
            # clic, et une réunion close n'encombre pas la bibliothèque.
            # Ce dont le modèle n'était pas sûr : l'app macOS l'écrivait
            # dans un fichier « à vérifier », le serveur le jetait.
            ("uncertain_json", "TEXT"),
            # Vidéo compressée, hors du serveur. La session d'envoi vaut
            # autorisation d'écrire : elle ne quitte jamais la base.
            # Ce qu'est devenu le dépôt automatique dans Odoo : la
            # finalisation se fait en tâche de fond, personne n'est là
            # pour lire la réponse sur le moment.
            ("odoo_depot_json", "TEXT"),
            ("meeting_date", "TEXT"),
            ("video_file_id", "TEXT"),
            ("video_bytes", "INTEGER"),
            ("video_session", "TEXT"),
            ("video_uploaded_at", "TEXT"),
            ("video_lectures", "INTEGER NOT NULL DEFAULT 0"),
            ("archived_at", "TEXT"),
            ("deleted_at", "TEXT"),
        ):
            self._ensure_column(conn, "jobs", colonne, ddl)

    def _ensure_column(self, conn, table: str, column: str, ddl: str) -> None:
        """Ajout de colonne idempotent — le pendant du helper de sync-hub.

        SQLite n'a pas d'``ADD COLUMN IF NOT EXISTS`` : on lit le schéma
        courant plutôt que d'avaler une exception, qui masquerait une
        vraie erreur de DDL."""
        existing = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        with conn:
            yield conn

    # -- utilisateurs --------------------------------------------------

    def user_id_for_email(self, email: str) -> int:
        """Crée l'utilisateur à la volée : Access a déjà fait le tri."""
        normalized = email.strip().lower()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE email = ?", (normalized,)
            ).fetchone()
            if row:
                return int(row["id"])
            cursor = conn.execute("INSERT INTO users (email) VALUES (?)", (normalized,))
            return int(cursor.lastrowid)

    def email_for_user(self, owner_id: int) -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT email FROM users WHERE id = ?", (owner_id,)
            ).fetchone()
            return str(row["email"]) if row else ""

    def set_odoo_credentials(
        self, owner_id: int, *, login: str, key_chiffree: str
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET odoo_login = ?, odoo_key_chiffree = ? WHERE id = ?",
                (login.strip(), key_chiffree, owner_id),
            )

    def odoo_credentials(self, owner_id: int) -> tuple[str, str]:
        """Identifiant et clé **chiffrée**. Le déchiffrement est ailleurs :
        la base ne doit jamais rendre un secret en clair."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT odoo_login, odoo_key_chiffree FROM users WHERE id = ?",
                (owner_id,),
            ).fetchone()
            if not row:
                return "", ""
            return str(row["odoo_login"] or ""), str(row["odoo_key_chiffree"] or "")

    def clear_odoo_credentials(self, owner_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET odoo_login = NULL, odoo_key_chiffree = NULL "
                "WHERE id = ?",
                (owner_id,),
            )

    # -- jobs ----------------------------------------------------------

    def create_job(
        self,
        *,
        owner_id: int,
        filename: str,
        duration_seconds: float,
        model: str,
        language: str,
        context: dict[str, Any],
        chunks: list[tuple[float, float]],
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO jobs (owner_id, filename, duration_seconds, model, "
                "language, chunk_count, context_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    owner_id,
                    filename,
                    float(duration_seconds),
                    model,
                    language,
                    len(chunks),
                    json.dumps(context, ensure_ascii=False),
                ),
            )
            job_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO job_chunks (job_id, idx, start_second, end_second) "
                "VALUES (?, ?, ?, ?)",
                [(job_id, i, start, end) for i, (start, end) in enumerate(chunks)],
            )
            return job_id

    def reprocess_job(
        self,
        job_id: int,
        *,
        duration_seconds: float,
        model: str,
        language: str,
        context: dict[str, Any],
        chunks: list[tuple[float, float]],
    ) -> None:
        """Repart de zéro sur une réunion qui n'a pas de transcription — une
        récupérée, ou une qui a échoué.

        Sur place plutôt qu'une réunion neuve : sa vidéo, sa date et ce qui
        dit d'où elle vient restent attachés, et le navigateur lit sa vidéo
        à la même adresse pendant tout le traitement. Le contexte s'ajoute
        à l'ancien, qui garde par exemple `recovered_from`."""
        job = self.get_job(job_id) or {}
        ancien = json.loads(job.get("context_json") or "{}")
        with self.connect() as conn:
            conn.execute("DELETE FROM job_chunks WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM transcription_segments WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM segments_fts WHERE job_id = ?", (job_id,))
            conn.execute(
                "UPDATE jobs SET duration_seconds = ?, model = ?, language = ?, "
                "chunk_count = ?, context_json = ?, status = 'en_attente', "
                "error_message = NULL, title = NULL, transcript = NULL, "
                "uncertain_json = NULL, updated_at = ? WHERE id = ?",
                (
                    float(duration_seconds),
                    model,
                    language,
                    len(chunks),
                    json.dumps({**ancien, **context}, ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                    job_id,
                ),
            )
            conn.executemany(
                "INSERT INTO job_chunks (job_id, idx, start_second, end_second) "
                "VALUES (?, ?, ?, ?)",
                [(job_id, i, start, end) for i, (start, end) in enumerate(chunks)],
            )

    def import_job(self, *, owner_id: int, payload: dict[str, Any]) -> tuple[int, bool]:
        """Reprend une réunion déjà transcrite ailleurs.

        Idempotent sur (propriétaire, nom de fichier, date de création) :
        une reprise de plusieurs années de bibliothèque ne réussit jamais
        du premier coup, il faut pouvoir relancer sans doubler.
        """
        nom = str(payload.get("filename") or "").strip() or "sans nom"
        cree = str(payload.get("created_at") or "").strip() or datetime.now().isoformat(timespec="seconds")

        with self.connect() as conn:
            existant = conn.execute(
                "SELECT id FROM jobs WHERE owner_id = ? AND filename = ? AND created_at = ?",
                (owner_id, nom, cree),
            ).fetchone()
            if existant:
                return int(existant["id"]), False

            cursor = conn.execute(
                "INSERT INTO jobs (owner_id, filename, duration_seconds, model, "
                "language, status, chunk_count, context_json, title, transcript, "
                "speaker_map_json, technical_terms_json, cloud_cost_usd, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'fr', 'termine', 1, '{}', ?, ?, ?, ?, ?, ?, ?)",
                (
                    owner_id,
                    nom,
                    float(payload.get("duration_seconds") or 0),
                    str(payload.get("model") or ""),
                    str(payload.get("title") or ""),
                    str(payload.get("transcript") or ""),
                    json.dumps(payload.get("speakers") or {}, ensure_ascii=False),
                    json.dumps(payload.get("technical_terms") or [], ensure_ascii=False),
                    float(payload.get("cost_usd") or 0),
                    cree,
                    cree,
                ),
            )
            job_id = int(cursor.lastrowid)

        segments = payload.get("segments") or []
        if segments:
            self.replace_segments(job_id, segments)
        termes = payload.get("technical_terms") or []
        if termes:
            # Le vocabulaire d'équipe hérite de tout l'historique : à la
            # bascule, les suggestions ne partent pas de zéro.
            self.record_vocabulary([str(t) for t in termes])
        return job_id, True

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    # États d'une réunion dans la bibliothèque. « actif » est le défaut :
    # une archive ou une corbeille qu'on voit par accident n'aide
    # personne.
    ETATS = {
        "actif": "deleted_at IS NULL AND archived_at IS NULL",
        "archive": "deleted_at IS NULL AND archived_at IS NOT NULL",
        "corbeille": "deleted_at IS NOT NULL",
        "tout": "1 = 1",
    }

    # Les colonnes par lesquelles la bibliothèque se trie. La date qui compte
    # est celle de la réunion ; à défaut, celle du dépôt. Une liste fermée :
    # le nom de colonne vient de la requête, il ne doit pas pouvoir être
    # autre chose.
    SORTS = {
        "date": "COALESCE(meeting_date, created_at)",
        "title": "COALESCE(NULLIF(title, ''), filename) COLLATE NOCASE",
        "duration": "duration_seconds",
        "status": "status",
        "cost": "cloud_cost_usd",
    }

    def list_jobs(
        self,
        owner_id: int,
        limit: int = 100,
        etat: str = "actif",
        *,
        sort: str = "date",
        descending: bool = True,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Une page de la bibliothèque, triée par le serveur.

        Trier côté navigateur ne marchait qu'en apparence : le serveur ne
        rendait que les 100 dernières réunions *créées*, et cent fichiers
        récupérés d'un coup suffisaient à faire disparaître tout le reste.
        """
        filtre = self.ETATS.get(etat, self.ETATS["actif"])
        ordre = self.SORTS.get(sort, self.SORTS["date"])
        sens = "DESC" if descending else "ASC"
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE owner_id = ? AND {filtre} "
                f"ORDER BY {ordre} {sens}, id {sens} LIMIT ? OFFSET ?",
                (owner_id, limit, max(0, offset)),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_jobs(self, owner_id: int, etat: str = "actif") -> int:
        filtre = self.ETATS.get(etat, self.ETATS["actif"])
        with self.connect() as conn:
            return int(conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE owner_id = ? AND {filtre}", (owner_id,)
            ).fetchone()[0])

    # -- corbeille et archives -------------------------------------------

    def _marquer(self, job_id: int, champ: str, valeur: str | None) -> None:
        with self.connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {champ} = ?, updated_at = ? WHERE id = ?",
                (valeur, datetime.now().isoformat(timespec="seconds"), job_id),
            )

    def jeter_job(self, job_id: int) -> None:
        """À la corbeille : réversible, et purgée après le délai."""
        self._marquer(job_id, "deleted_at", datetime.now().isoformat(timespec="seconds"))

    def archiver_job(self, job_id: int) -> None:
        """Hors de la bibliothèque, mais intacte et cherchable."""
        self._marquer(job_id, "archived_at", datetime.now().isoformat(timespec="seconds"))

    def restaurer_job(self, job_id: int) -> None:
        """Ressort d'archive comme de corbeille : un seul geste à retenir."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET deleted_at = NULL, archived_at = NULL, "
                "updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), job_id),
            )

    def supprimer_job(self, job_id: int) -> None:
        """Suppression définitive, index de recherche compris.

        `ON DELETE CASCADE` emporte les fenêtres et les segments, mais
        pas la table FTS, qui est virtuelle et ignore les clés
        étrangères : l'oublier laisserait la réunion trouvable après sa
        suppression."""
        with self.connect() as conn:
            conn.execute("DELETE FROM segments_fts WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def ouvrir_envoi_video(self, job_id: int, session: str, taille: int) -> None:
        """La vidéo déjà là, s'il y en a une, reste lisible jusqu'à ce que
        la nouvelle soit entière : un envoi abandonné ne la perd pas."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET video_session = ?, video_bytes = ? WHERE id = ?",
                (session, int(taille), job_id),
            )

    def terminer_envoi_video(self, job_id: int, fichier_id: str) -> str | None:
        """Rend l'identifiant de la vidéo remplacée, à effacer du stockage."""
        with self.connect() as conn:
            ancienne = conn.execute(
                "SELECT video_file_id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            conn.execute(
                "UPDATE jobs SET video_file_id = ?, video_session = NULL, "
                "video_uploaded_at = ? WHERE id = ?",
                (fichier_id, datetime.now().isoformat(timespec="seconds"), job_id),
            )
        precedente = ancienne["video_file_id"] if ancienne else None
        return precedente if precedente and precedente != fichier_id else None

    def compter_lecture_video(self, job_id: int) -> None:
        """Chaque lecture est comptée : en stockage froid, c'est elle qui
        coûte. Le chiffre servira le jour où GCS la facturera vraiment."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET video_lectures = video_lectures + 1 WHERE id = ?",
                (job_id,),
            )

    def corbeille_perimee(self, jours: int) -> list[int]:
        """Les réunions jetées depuis plus de ``jours`` jours."""
        if jours <= 0:
            return []
        limite = (datetime.now() - timedelta(days=jours)).isoformat(timespec="seconds")
        with self.connect() as conn:
            return [
                int(r["id"])
                for r in conn.execute(
                    "SELECT id FROM jobs WHERE deleted_at IS NOT NULL AND deleted_at < ?",
                    (limite,),
                )
            ]

    def purger_corbeille(self, jours: int) -> list[int]:
        """Vide ce qui a dépassé le délai de rétention. Rend les
        identifiants supprimés, pour que ça se journalise."""
        perimes = self.corbeille_perimee(jours)
        for job_id in perimes:
            self.supprimer_job(job_id)
        return perimes

    def renommer_interlocuteurs(self, job_id: int, carte: dict[str, str]) -> dict[str, str]:
        """Applique les noms choisis aux répliques, pas seulement à la carte.

        Enregistrer « Intervenant 2 → Steeve Ouinet » ne changeait que la
        table de correspondance : les répliques gardaient leur étiquette,
        et rien ne semblait s'être passé. Les segments, l'index de
        recherche et le texte sont réécrits ; la carte redevient
        l'identité des noms actuels, pour qu'un second renommage parte de
        ce qu'on voit.

        Le texte est corrigé en place plutôt que régénéré : celui des
        réunions reprises du Mac ne vient pas des segments.
        """
        renommages = {
            ancien: nouveau.strip()
            for ancien, nouveau in (carte or {}).items()
            if nouveau and nouveau.strip() and nouveau.strip() != ancien
        }
        job = self.get_job(job_id) or {}
        if renommages:
            segments = [
                {"start": s["start_second"], "end": s["end_second"],
                 "speaker": renommages.get(s["speaker"] or "", s["speaker"] or ""),
                 "text": s["text"]}
                for s in self.segments_for_job(job_id)
            ]
            self.replace_segments(job_id, segments)
            texte = job.get("transcript") or ""
            for ancien, nouveau in renommages.items():
                # En tête de ligne, après un éventuel horodatage entre
                # crochets : « Intervenant 2 : » comme « [00:12] Intervenant 2: ».
                motif = re.compile(
                    r"(^|\n)((?:\[[^\]\n]*\]\s*)?)" + re.escape(ancien) + r"(\s*:)"
                )
                texte = motif.sub(lambda m: f"{m.group(1)}{m.group(2)}{nouveau}{m.group(3)}", texte)
            with self.connect() as conn:
                conn.execute("UPDATE jobs SET transcript = ? WHERE id = ?", (texte, job_id))
        noms = [
            (nouveau.strip() or ancien)
            for ancien, nouveau in (carte or {}).items()
        ]
        nouvelle_carte = {nom: nom for nom in dict.fromkeys(noms)}
        self.update_job_context(job_id, speakers=nouvelle_carte)
        return nouvelle_carte

    def set_meeting_date(self, job_id: int, date_iso: str | None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET meeting_date = ?, updated_at = ? WHERE id = ?",
                (date_iso, datetime.now().isoformat(timespec="seconds"), job_id),
            )

    def noter_depot_odoo(self, job_id: int, issue: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET odoo_depot_json = ? WHERE id = ?",
                (json.dumps(issue or {}, ensure_ascii=False), job_id),
            )

    def reserver_finalisation(self, job_id: int) -> bool:
        """Réserve la fusion finale, atomiquement.

        Vrai pour le premier qui demande, faux pour les suivants : deux
        fenêtres qui terminent ensemble ne doivent pas fusionner deux
        fois (et déposer deux notes dans Odoo)."""
        with self.connect() as conn:
            curseur = conn.execute(
                "UPDATE jobs SET status = 'finalisation', updated_at = ? "
                "WHERE id = ? AND status NOT IN ('finalisation', 'termine')",
                (datetime.now().isoformat(timespec="seconds"), job_id),
            )
            return curseur.rowcount == 1

    def set_job_status(
        self, job_id: int, status: str, *, error: str | None = None
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error_message = ?, updated_at = ? "
                "WHERE id = ?",
                (status, error, datetime.now().isoformat(timespec="seconds"), job_id),
            )

    def finish_job(
        self,
        job_id: int,
        *,
        title: str,
        transcript: str,
        speakers: dict[str, str],
        technical_terms: list[str],
        uncertain: list[dict[str, Any]] | None = None,
        cost_usd: float,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'termine', title = ?, transcript = ?, "
                "speaker_map_json = ?, technical_terms_json = ?, "
                "uncertain_json = ?, cloud_cost_usd = ?, "
                "error_message = NULL, updated_at = ? WHERE id = ?",
                (
                    title,
                    transcript,
                    json.dumps(speakers, ensure_ascii=False),
                    json.dumps(technical_terms, ensure_ascii=False),
                    json.dumps(uncertain or [], ensure_ascii=False),
                    float(cost_usd),
                    datetime.now().isoformat(timespec="seconds"),
                    job_id,
                ),
            )

    # -- segments (fenêtres) -------------------------------------------

    def chunks_for_job(self, job_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM job_chunks WHERE job_id = ? ORDER BY idx", (job_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def set_chunk_status(
        self,
        job_id: int,
        idx: int,
        status: str,
        *,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE job_chunks SET status = ?, error = ?, result_json = "
                "COALESCE(?, result_json), updated_at = ? WHERE job_id = ? AND idx = ?",
                (
                    status,
                    error,
                    json.dumps(result, ensure_ascii=False) if result else None,
                    datetime.now().isoformat(timespec="seconds"),
                    job_id,
                    idx,
                ),
            )

    def replace_segments(self, job_id: int, segments: list[dict[str, Any]]) -> None:
        rows = [
            (
                job_id,
                float(s.get("start") or 0.0),
                float(s.get("end") or 0.0),
                str(s.get("speaker") or ""),
                str(s.get("text") or ""),
            )
            for s in segments
        ]
        with self.connect() as conn:
            conn.execute("DELETE FROM transcription_segments WHERE job_id = ?", (job_id,))
            conn.executemany(
                "INSERT INTO transcription_segments (job_id, start_second, "
                "end_second, speaker, text) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            # L'index plein texte suit dans la même transaction : une
            # recherche ne doit jamais ramener une phrase qui n'existe plus.
            conn.execute("DELETE FROM segments_fts WHERE job_id = ?", (job_id,))
            conn.executemany(
                "INSERT INTO segments_fts (text, job_id, start_second, speaker) "
                "VALUES (?, ?, ?, ?)",
                [(text, job_id, start, speaker) for _, start, _, speaker, text in rows],
            )

    def segments_for_job(self, job_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT start_second, end_second, speaker, text FROM "
                "transcription_segments WHERE job_id = ? ORDER BY start_second",
                (job_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def reset_chunk(self, job_id: int, idx: int) -> None:
        """Remet une fenêtre à l'état attendu pour que le navigateur la
        réencode. C'est le mécanisme derrière « relancer cette fenêtre » :
        sur une réunion de huit fenêtres dont trois ont échoué, rien ne
        justifie de repayer les cinq qui sont passées."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE job_chunks SET status = 'attendu', error = NULL, "
                "result_json = NULL, updated_at = ? WHERE job_id = ? AND idx = ?",
                (datetime.now().isoformat(timespec="seconds"), job_id, idx),
            )

    def archive_current_version(self, job_id: int) -> None:
        """Empile la transcription en place dans l'historique avant qu'une
        relance ne l'écrase. Sans cela, relancer une réunion pour corriger
        une fenêtre détruirait le travail déjà validé dessus."""
        job = self.get_job(job_id)
        if not job or not (job.get("transcript") or "").strip():
            return
        versions = json.loads(job.get("previous_versions_json") or "[]")
        versions.insert(
            0,
            {
                "archived_at": datetime.now().isoformat(timespec="seconds"),
                "title": job.get("title") or "",
                "transcript": job.get("transcript") or "",
                "speaker_map_json": job.get("speaker_map_json"),
                "technical_terms_json": job.get("technical_terms_json"),
                "cloud_cost_usd": job.get("cloud_cost_usd") or 0.0,
                # Les répliques horodatées : sans elles, une version se lit
                # mais ne se restaure qu'à condition d'avoir la même forme
                # que le texte en place.
                "segments": [
                    {"start": s["start_second"], "end": s["end_second"],
                     "speaker": s["speaker"], "text": s["text"]}
                    for s in self.segments_for_job(job_id)
                ],
            },
        )
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET previous_versions_json = ?, updated_at = ? WHERE id = ?",
                (
                    json.dumps(versions[:10], ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                    job_id,
                ),
            )

    @staticmethod
    def repliques_de_version(
        version: dict[str, Any], en_place: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Les répliques d'une version, ou ``None`` si elles sont perdues.

        Les versions d'avant n'ont gardé que le texte « Nom : réplique ».
        Une relecture ne change que les noms et les mots, pas le découpage :
        quand le texte a autant de lignes que la transcription en place a
        de répliques, les horodatages en place lui vont.
        """
        if version.get("segments"):
            return list(version["segments"])
        lignes = [l for l in (version.get("transcript") or "").splitlines() if l.strip()]
        if not lignes or len(lignes) != len(en_place):
            return None
        repliques = []
        for ligne, seg in zip(lignes, en_place):
            nom, separateur, texte = ligne.partition(" : ")
            if not separateur:
                nom, texte = "", ligne
            repliques.append({"start": seg["start_second"], "end": seg["end_second"],
                              "speaker": nom.strip(), "text": texte.strip()})
        return repliques

    def restaurer_version(self, job_id: int, rang: int) -> dict[str, Any]:
        """Remet une version en place. Celle qu'elle remplace rejoint
        l'historique : revenir en arrière se défait de la même façon."""
        job = self.get_job(job_id)
        versions = json.loads((job or {}).get("previous_versions_json") or "[]")
        if not job or not 0 <= rang < len(versions):
            raise KeyError(rang)
        version = versions[rang]
        repliques = self.repliques_de_version(version, self.segments_for_job(job_id))
        if repliques is None:
            raise ValueError("Cette version n'a gardé que son texte.")
        self.archive_current_version(job_id)
        # La version restaurée quitte l'historique : elle est en place, la
        # garder aussi dans la liste la montrerait deux fois.
        versions = json.loads(self.get_job(job_id).get("previous_versions_json") or "[]")
        del versions[rang + 1]
        maintenant = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET title = ?, transcript = ?, speaker_map_json = ?, "
                "technical_terms_json = ?, previous_versions_json = ?, updated_at = ? "
                "WHERE id = ?",
                (
                    version.get("title") or "",
                    version.get("transcript") or "",
                    version.get("speaker_map_json") or "{}",
                    version.get("technical_terms_json") or "[]",
                    json.dumps(versions, ensure_ascii=False),
                    maintenant,
                    job_id,
                ),
            )
        self.replace_segments(job_id, repliques)
        return version

    def update_job_context_json(self, job_id: int, contexte: dict[str, Any]) -> None:
        """Le contexte retenu pour cette réunion, après coup.

        Recoller un autre dossier Odoo change le contexte : le garder
        permet de relancer un enrichissement sans redemander Odoo."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET context_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(contexte, ensure_ascii=False),
                 datetime.now().isoformat(timespec="seconds"), job_id),
            )

    def set_uncertain(self, job_id: int, passages: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET uncertain_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(passages or [], ensure_ascii=False),
                 datetime.now().isoformat(timespec="seconds"), job_id),
            )

    def update_job_context(
        self,
        job_id: int,
        *,
        title: str | None = None,
        speakers: dict[str, str] | None = None,
        technical_terms: list[str] | None = None,
    ) -> None:
        """Édition de la fiche par l'utilisateur. Chaque champ absent est
        laissé tel quel — un formulaire partiel ne doit pas effacer ce
        qu'il n'affichait pas."""
        sets: list[str] = []
        values: list[Any] = []
        if title is not None:
            sets.append("title = ?")
            values.append(title.strip())
        if speakers is not None:
            sets.append("speaker_map_json = ?")
            values.append(json.dumps(speakers, ensure_ascii=False))
        if technical_terms is not None:
            sets.append("technical_terms_json = ?")
            values.append(json.dumps(technical_terms, ensure_ascii=False))
        if not sets:
            return
        sets.append("updated_at = ?")
        values.extend([datetime.now().isoformat(timespec="seconds"), job_id])
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", values)

    def set_transcript(self, job_id: int, transcript: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET transcript = ?, updated_at = ? WHERE id = ?",
                (transcript, datetime.now().isoformat(timespec="seconds"), job_id),
            )

    def search_segments(
        self, owner_id: int, query: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Recherche plein texte dans les transcriptions de l'utilisateur.

        La requête est découpée en termes, chacun cité, pour qu'une
        apostrophe ou un tiret ne soit pas lu comme de la syntaxe FTS —
        « l'équipe » ferait sinon une erreur d'analyse plutôt qu'une
        recherche.
        """
        terms = [t for t in re.split(r"\s+", (query or "").strip()) if t]
        if not terms:
            return []
        match = " ".join('"' + t.replace('"', '""') + '"' for t in terms)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT f.job_id, f.start_second, f.speaker, f.text, "
                "       j.title, j.filename "
                "FROM segments_fts f JOIN jobs j ON j.id = f.job_id "
                # Une réunion à la corbeille ne doit plus ressortir ;
                # une archive, si — c'est tout l'intérêt d'archiver
                # plutôt que de jeter.
                "WHERE segments_fts MATCH ? AND j.owner_id = ? "
                "  AND j.deleted_at IS NULL "
                "ORDER BY rank LIMIT ?",
                (match, owner_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- onboarding ------------------------------------------------------

    def onboarding_state(self, user_id: int) -> dict[str, bool]:
        """Où en est la personne de ses premiers pas : chaque étape se lit
        dans ce qu'elle a fait, pas dans une case qu'elle aurait cochée."""
        with self.connect() as conn:
            user = conn.execute(
                "SELECT odoo_login, odoo_key_chiffree, onboarding_dismissed_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            mac = conn.execute(
                "SELECT 1 FROM enrolements WHERE owner_id = ? LIMIT 1", (user_id,)
            ).fetchone()
            job = conn.execute("SELECT 1 FROM jobs WHERE owner_id = ? LIMIT 1", (user_id,)).fetchone()
        return {
            "odoo": bool(user and user["odoo_login"] and user["odoo_key_chiffree"]),
            "mac": bool(mac),
            "transcription": bool(job),
            "dismissed": bool(user and user["onboarding_dismissed_at"]),
        }

    def dismiss_onboarding(self, user_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET onboarding_dismissed_at = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), user_id),
            )

    # -- Google account and history recovery ----------------------------

    def save_google_account(self, user_id: int, email: str, refresh_token_enc: str, scope: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO google_accounts (user_id, email, refresh_token_enc, scope, connected_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET email = excluded.email, "
                "refresh_token_enc = excluded.refresh_token_enc, scope = excluded.scope, "
                "connected_at = excluded.connected_at",
                (user_id, email, refresh_token_enc, scope, datetime.now().isoformat(timespec="seconds")),
            )

    def google_account(self, user_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM google_accounts WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def delete_google_account(self, user_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM google_accounts WHERE user_id = ?", (user_id,))

    def save_oauth_state(self, state: str, user_id: int, verifier: str) -> None:
        maintenant = datetime.now()
        with self.connect() as conn:
            # Les demandes abandonnées ne s'accumulent pas.
            conn.execute(
                "DELETE FROM oauth_states WHERE created_at < ?",
                ((maintenant - timedelta(minutes=15)).isoformat(timespec="seconds"),),
            )
            conn.execute(
                "INSERT INTO oauth_states (state, user_id, verifier, created_at) VALUES (?, ?, ?, ?)",
                (state, user_id, verifier, maintenant.isoformat(timespec="seconds")),
            )

    def pop_oauth_state(self, state: str, user_id: int) -> str | None:
        """Le vérificateur d'une demande en cours, consommé : un état ne sert
        qu'une fois, et seulement à celui qui l'a demandé, dans le quart
        d'heure."""
        limite = (datetime.now() - timedelta(minutes=15)).isoformat(timespec="seconds")
        with self.connect() as conn:
            row = conn.execute(
                "SELECT verifier FROM oauth_states WHERE state = ? AND user_id = ? AND created_at >= ?",
                (state, user_id, limite),
            ).fetchone()
            conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
        return row["verifier"] if row else None

    def dismiss_recovery(self, user_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET recovery_dismissed_at = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), user_id),
            )

    def recovery_dismissed(self, user_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT recovery_dismissed_at FROM users WHERE id = ?", (user_id,)).fetchone()
        return bool(row and row["recovery_dismissed_at"])

    def recovered_sources(self, source: str) -> dict[str, dict[str, Any]]:
        """Ce qui a déjà été récupéré depuis une source, par identifiant
        externe, avec l'adresse de qui l'a fait."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT r.*, u.email AS user_email FROM recovered_sources r "
                "JOIN users u ON u.id = r.user_id WHERE r.source = ?",
                (source,),
            ).fetchall()
        return {row["external_id"]: dict(row) for row in rows}

    def recovered_checksums(self) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT r.*, u.email AS user_email FROM recovered_sources r "
                "JOIN users u ON u.id = r.user_id WHERE r.checksum IS NOT NULL",
            ).fetchall()
        return {row["checksum"]: dict(row) for row in rows}

    def record_recovered(
        self, *, source: str, external_id: str, checksum: str | None, size_bytes: int | None,
        name: str, job_id: int | None, user_id: int,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO recovered_sources (source, external_id, checksum, size_bytes, name, job_id, "
                "user_id, recovered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (source, external_id, checksum, size_bytes, name, job_id, user_id,
                 datetime.now().isoformat(timespec="seconds")),
            )
            return int(cur.lastrowid)

    def mark_original_trashed(self, source: str, external_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE recovered_sources SET original_trashed_at = ? WHERE source = ? AND external_id = ?",
                (datetime.now().isoformat(timespec="seconds"), source, external_id),
            )

    def library_fingerprints(self) -> list[dict[str, Any]]:
        """Nom de fichier, durée et date de toutes les réunions, tous comptes
        confondus : de quoi reconnaître un enregistrement déjà transcrit
        sous un autre chemin (l'app macOS, un envoi à la main)."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, owner_id, filename, duration_seconds, video_bytes, "
                "COALESCE(meeting_date, created_at) AS quand FROM jobs"
            ).fetchall()
        return [dict(r) for r in rows]

    def set_odoo_link(
        self, job_id: int, *, model: str, record_id: int, message_id: int | None
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET odoo_model = ?, odoo_record_id = ?, "
                "odoo_message_id = ?, odoo_published_at = ?, updated_at = ? "
                "WHERE id = ?",
                (
                    model,
                    record_id,
                    message_id,
                    datetime.now().isoformat(timespec="seconds") if message_id else None,
                    datetime.now().isoformat(timespec="seconds"),
                    job_id,
                ),
            )

    # -- jetons d'API ---------------------------------------------------

    def create_api_token(self, owner_id: int, name: str) -> tuple[int, str]:
        """Fabrique un jeton et n'en garde que l'empreinte.

        Renvoie ``(id, jeton en clair)`` — la seule fois où la valeur
        existe. L'appelant doit la montrer puis l'oublier.
        """
        secret = "ekt_" + secrets.token_urlsafe(32)
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO api_tokens (owner_id, name, token_sha256) "
                "VALUES (?, ?, ?)",
                (owner_id, (name or "").strip() or "sans nom", _digest(secret)),
            )
            return int(cursor.lastrowid), secret

    # -- enrôlement d'appareil --------------------------------------------

    # Alphabet sans O/0/I/1 : le code se lit à voix haute et se recopie
    # depuis une fenêtre d'application.
    ALPHABET_CODE = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

    def ouvrir_enrolement(
        self, appareil: str, *, minutes: int = 15
    ) -> tuple[str, str, str]:
        """Ouvre une demande et rend (code appareil, code humain, échéance).

        Le code appareil est un secret que seule l'app connaît ; le code
        humain s'affiche pour être reconnu dans le navigateur. Les deux
        expirent vite : une demande oubliée ne doit pas rester ouverte.
        """
        code_appareil = "ekd_" + secrets.token_urlsafe(32)
        with self.connect() as conn:
            for _ in range(10):
                code_humain = "-".join(
                    "".join(secrets.choice(self.ALPHABET_CODE) for _ in range(4))
                    for _ in range(2)
                )
                if not conn.execute(
                    "SELECT 1 FROM enrolements WHERE code_humain = ?", (code_humain,)
                ).fetchone():
                    break
            else:  # pragma: no cover - 32^8 collisions d'affilée
                raise RuntimeError("Impossible de tirer un code d'enrôlement libre.")
            echeance = (datetime.now() + timedelta(minutes=minutes)).isoformat(
                timespec="seconds"
            )
            conn.execute(
                "INSERT INTO enrolements (appareil, code_appareil_sha, code_humain, "
                "expires_at) VALUES (?, ?, ?, ?)",
                ((appareil or "").strip()[:120] or "appareil inconnu",
                 _digest(code_appareil), code_humain, echeance),
            )
        return code_appareil, code_humain, echeance

    def enrolement_par_code_humain(self, code: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            ligne = conn.execute(
                "SELECT * FROM enrolements WHERE code_humain = ?",
                ((code or "").strip().upper(),),
            ).fetchone()
        return dict(ligne) if ligne else None

    def approuver_enrolement(self, code: str, owner_id: int) -> bool:
        """Valide une demande. Faux si elle est inconnue, expirée ou déjà
        traitée — dans les trois cas, l'appareil doit recommencer."""
        demande = self.enrolement_par_code_humain(code)
        if not demande or demande["statut"] != "en_attente":
            return False
        if demande["expires_at"] < datetime.now().isoformat(timespec="seconds"):
            return False
        with self.connect() as conn:
            conn.execute(
                "UPDATE enrolements SET statut = 'approuve', owner_id = ? WHERE id = ?",
                (owner_id, demande["id"]),
            )
        return True

    def reclamer_enrolement(self, code_appareil: str) -> dict[str, Any]:
        """L'appareil vient chercher son jeton.

        Rend ``{"statut": ...}`` et, une seule fois, le jeton. La demande
        est close dans la foulée : un code appareil rejoué ne redonne
        jamais un second jeton.
        """
        with self.connect() as conn:
            ligne = conn.execute(
                "SELECT * FROM enrolements WHERE code_appareil_sha = ?",
                (_digest(code_appareil),),
            ).fetchone()
        if ligne is None:
            return {"statut": "inconnu"}
        demande = dict(ligne)
        if demande["statut"] != "approuve":
            if demande["expires_at"] < datetime.now().isoformat(timespec="seconds"):
                return {"statut": "expire"}
            return {"statut": demande["statut"]}
        token_id, secret = self.create_api_token(
            int(demande["owner_id"]), f"appareil — {demande['appareil']}"
        )
        with self.connect() as conn:
            conn.execute(
                "UPDATE enrolements SET statut = 'consomme' WHERE id = ?",
                (demande["id"],),
            )
        return {"statut": "approuve", "token": secret, "token_id": token_id,
                "email": self.email_for_user(int(demande["owner_id"]))}

    def purger_enrolements(self) -> int:
        """Les demandes mortes ne servent plus qu'à encombrer."""
        with self.connect() as conn:
            curseur = conn.execute(
                "DELETE FROM enrolements WHERE statut = 'consomme' OR expires_at < ?",
                (datetime.now().isoformat(timespec="seconds"),),
            )
            return curseur.rowcount or 0

    def owner_for_api_token(self, secret: str) -> int | None:
        """Propriétaire d'un jeton valide, ou None.

        La recherche se fait sur l'empreinte : le jeton en clair ne
        touche jamais la base, et l'index porte donc sur une valeur de
        longueur fixe, insensible au contenu du jeton.
        """
        raw = (secret or "").strip()
        if not raw:
            return None
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, owner_id FROM api_tokens "
                "WHERE token_sha256 = ? AND revoked_at IS NULL",
                (_digest(raw),),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE api_tokens SET last_used_at = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), row["id"]),
            )
            return int(row["owner_id"])

    def list_api_tokens(self, owner_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, name, created_at, last_used_at, revoked_at "
                "FROM api_tokens WHERE owner_id = ? ORDER BY id DESC",
                (owner_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def revoke_api_token(self, owner_id: int, token_id: int) -> bool:
        """Révoque sans supprimer : la ligne garde la trace du dernier
        usage, qui est précisément ce qu'on veut consulter après coup."""
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE api_tokens SET revoked_at = ? "
                "WHERE id = ? AND owner_id = ? AND revoked_at IS NULL",
                (datetime.now().isoformat(timespec="seconds"), token_id, owner_id),
            )
            return cursor.rowcount > 0

    # -- vocabulaire ---------------------------------------------------

    def record_vocabulary(self, terms: list[str]) -> None:
        """Enregistre les termes d'un traitement : usage + appariements.

        Les paires sont écrites symétriquement, pour qu'une suggestion
        fonctionne quel que soit le terme saisi en premier.
        """
        propres = []
        vus: set[str] = set()
        for brut in terms:
            terme = unicodedata.normalize("NFC", str(brut or "").strip())
            cle = terme.lower()
            if terme and cle not in vus:
                vus.add(cle)
                propres.append(terme)
        if not propres:
            return

        maintenant = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            for terme in propres:
                conn.execute(
                    "INSERT INTO vocabulary (term, usages, last_used) VALUES (?, 1, ?) "
                    "ON CONFLICT(term) DO UPDATE SET usages = usages + 1, last_used = ?",
                    (terme, maintenant, maintenant),
                )
            for terme in propres:
                for pair in propres:
                    if terme == pair:
                        continue
                    conn.execute(
                        "INSERT INTO vocabulary_pairs (term, peer, count) VALUES (?, ?, 1) "
                        "ON CONFLICT(term, peer) DO UPDATE SET count = count + 1",
                        (terme, pair),
                    )

    def suggest_vocabulary(
        self, selected: list[str], limit: int = 30
    ) -> list[dict[str, Any]]:
        """Termes à proposer, les plus pertinents d'abord.

        Le tri place la **co-occurrence avant l'usage brut** : c'est ce
        qui fait remonter un terme rare mais lié à ce qui est déjà
        sélectionné, plutôt que les mêmes cinq termes omniprésents à
        chaque réunion.
        """
        deja = {unicodedata.normalize("NFC", t.strip()).lower() for t in selected if t.strip()}
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT v.term, v.usages, COALESCE(SUM(p.count), 0) AS affinite "
                "FROM vocabulary v "
                "LEFT JOIN vocabulary_pairs p "
                "  ON p.term = v.term AND LOWER(p.peer) IN "
                f"     ({','.join('?' * len(deja)) or 'NULL'}) "
                "GROUP BY v.term "
                "ORDER BY affinite DESC, v.usages DESC, v.term COLLATE NOCASE",
                tuple(deja),
            ).fetchall()
        return [
            {"term": r["term"], "usages": r["usages"], "affinity": r["affinite"]}
            for r in rows
            if r["term"].lower() not in deja
        ][:limit]

    def forget_vocabulary(self, term: str) -> None:
        cible = unicodedata.normalize("NFC", str(term or "").strip())
        with self.connect() as conn:
            conn.execute("DELETE FROM vocabulary WHERE term = ?", (cible,))
            conn.execute(
                "DELETE FROM vocabulary_pairs WHERE term = ? OR peer = ?", (cible, cible)
            )

    # -- dépenses ------------------------------------------------------

    def add_api_usage(
        self,
        *,
        job_id: int | None,
        provider: str,
        model: str,
        step: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO api_usage (job_id, provider, model, step, "
                "input_tokens, output_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    provider,
                    model,
                    step,
                    int(input_tokens),
                    int(output_tokens),
                    float(cost_usd),
                ),
            )

    def month_spend_usd(self, month: str | None = None) -> float:
        """Dépense du mois — plafond d'équipe, la clé Gemini étant partagée."""
        period = (month or datetime.now().strftime("%Y-%m")).strip()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM api_usage "
                "WHERE strftime('%Y-%m', created_at) = ?",
                (period,),
            ).fetchone()
            return float(row["total"] if row else 0.0)
