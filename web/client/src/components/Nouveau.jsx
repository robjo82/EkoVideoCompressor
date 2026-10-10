import { useEffect, useRef, useState } from 'react';
import { Bouton, Champ, DateHeure, Erreur } from './Communs.jsx';
import { sonderDuree, identifier } from '../sonde.js';
import { api } from '../api.js';
import { useCompression } from '../useCompression.js';
import { archiver, useArchivage } from '../archivage.js';
import { Apercu } from './Apercu.jsx';
import { Sonde } from './Sonde.jsx';
import { interrompre, reprendre, usePipeline } from '../usePipeline.js';
import {
  consumeLaunchRequest, enqueue, isMedia, removeFromQueue, takeNext, useFileDrag, useFileQueue,
} from '../fileQueue.js';
import { duree, mo, usd, horodatage, liste } from '../format.js';

const MODELE = 'gemini-3.8-flash';

/** Assistant de lancement.
 *
 *  L'intention : rendre lisible et rassurant un traitement long. D'où
 *  une seule colonne, trois moments — le fichier, le contexte, puis
 *  l'avancement — et une phrase qui dit franchement que le média ne
 *  quitte pas le poste. C'est la promesse centrale de l'outil ; elle
 *  mérite d'être écrite, pas déduite.
 */
/** « 2026-09-21T18:34 » pour un champ datetime-local, à l'heure locale. */
function versChampLocal(date) {
  const decale = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return decale.toISOString().slice(0, 16);
}

/** Au-delà de ce débit, la vidéo stockée d'une réunion récupérée n'a
 *  jamais été compressée : la retraiter propose de la remplacer par sa
 *  version légère (environ 0,2 Mbit/s). */
const DEBIT_A_COMPRESSER = 1_000_000;

/** Au-dessous de ce poids, le fichier est copié en mémoire dès qu'on le
 *  choisit. Un mémo vocal synchronisé par iCloud peut être réécrit sur le
 *  disque entre le dépôt et l'encodage ; Chrome refuse alors de le relire,
 *  avec un « network error » qui n'a rien de réseau. La copie, elle, ne
 *  bouge plus. Au-delà, une vidéo de plusieurs gigaoctets pèserait trop. */
const COPIE_MAX = 400 * 1024 * 1024;

async function copieEnMemoire(fichier) {
  // Une réunion qu'on retraite n'est pas sur ce poste : elle se relit
  // depuis le stockage, qui ne bouge pas.
  if (fichier.remote || fichier.size > COPIE_MAX) return fichier;
  const octets = await fichier.arrayBuffer();
  return new File([octets], fichier.name, { type: fichier.type, lastModified: fichier.lastModified });
}

export function Nouveau({ surTermine, surBibliotheque }) {
  const [fichier, setFichier] = useState(null);
  const [secondes, setSecondes] = useState(0);
  const [lecture, setLecture] = useState('');
  const [client, setClient] = useState('');
  const [participants, setParticipants] = useState('');
  const [glossaire, setGlossaire] = useState('');
  const [contexteOdoo, setContexteOdoo] = useState('');
  const [mode, setMode] = useState('transcrire');
  const [videoDisponible, setVideoDisponible] = useState(false);
  // Quand la réunion a eu lieu. Proposée d'après le fichier — daté de la
  // fin de l'enregistrement, d'où la durée retranchée — et corrigeable :
  // un fichier recopié ou renommé porte une date qui ment.
  const [dateReunion, setDateReunion] = useState('');
  // L'archivage démarre avant que le traitement existe : la compression
  // n'attend pas la création du job, l'envoi si.
  const cibleRef = useRef(null);
  // Le mode choisi pour les fichiers du poste, qu'une réunion retraitée
  // ne doit pas changer pour les suivants.
  const modeChoisi = useRef(null);
  // Le champ fichier garde sa sélection tant qu'on ne le vide pas : sans
  // cela, « Lancer une autre transcription » montrait encore l'ancien.
  const champFichier = useRef(null);
  const archivage = useArchivage(cibleRef.current?.cle);
  const [sonde, setSonde] = useState(null);
  const [dossier, setDossier] = useState(null);
  // Une sonde lancée pour un fichier ne doit pas répondre pour le
  // suivant : on ne garde que la dernière.
  const sondeCourante = useRef(0);
  // Ce que le dossier retenu a versé dans le formulaire, pour le
  // reprendre s'il est retiré ou remplacé — sans toucher à ce que la
  // personne a saisi elle-même.
  const apport = useRef({ client: '', termes: [] });
  const [debut, setDebut] = useState(0);
  const [fin, setFin] = useState(0);
  const pipeline = usePipeline();
  const compression = useCompression();
  const enAttente = useFileQueue();
  const glisse = useFileDrag();
  const [surZone, setSurZone] = useState(false);
  const [ignores, setIgnores] = useState(0);

  const capacites = typeof AudioEncoder !== 'undefined' && window.isSecureContext;

  useEffect(() => {
    if (pipeline.etat === 'termine' && pipeline.resultat) surTermine?.(pipeline.jobId);
  }, [pipeline.etat, pipeline.resultat, pipeline.jobId, surTermine]);

  useEffect(() => {
    api.settings().then((r) => setVideoDisponible(Boolean(r.video?.disponible))).catch(() => {});
  }, []);

  useEffect(() => {
    const cible = cibleRef.current;
    if (!cible) return;
    if (pipeline.jobId != null) cible.jobId = pipeline.jobId;
    if (pipeline.etat === 'erreur') cible.abandon = true;
  }, [pipeline.jobId, pipeline.etat]);

  // Entrée de rodage : « ?source=/chemin » charge un fichier servi par le
  // serveur au lieu de passer par le sélecteur, ce qui rend la chaîne
  // vérifiable de bout en bout sans intervention. Même origine, derrière
  // Access comme le reste — rien n'est contourné.
  useEffect(() => {
    const source = new URLSearchParams(window.location.search).get('source');
    if (!source) return;
    (async () => {
      setLecture(`Chargement de ${source}…`);
      const blob = await (await fetch(source)).blob();
      const charge = new File([blob], source.split('/').pop(), { type: blob.type });
      const total = await sonderDuree(charge);
      setFichier(charge);
      setSecondes(total);
      setDebut(0);
      setFin(total);
      setDateReunion(versChampLocal(new Date(Date.now() - total * 1000)));
      setLecture(`${mo(charge.size)} · ${duree(total)} (rodage)`);
      // Même parcours qu'un fichier choisi à la main, sonde comprise.
      ecouter(charge, total);
    })().catch((e) => setLecture(`Rodage impossible : ${e.message}`));
  }, []);

  /** Écoute le début du fichier pour proposer un dossier Odoo.
   *
   *  Lancée d'office : elle coûte une fraction de centime, et sans elle
   *  il faut savoir soi-même quel dossier chercher. Son échec ne se
   *  remonte pas comme une erreur — on retombe simplement sur la
   *  saisie à la main.
   */
  async function ecouter(choisi, total, instant = '') {
    const numero = ++sondeCourante.current;
    const actuelle = () => numero === sondeCourante.current;
    setSonde({ enCours: true, phase: 'extrait', moment: instant });
    try {
      const reglages = await api.settings();
      const vue = await identifier(choisi, {
        audio: reglages.audio,
        fenetre: reglages.probe?.window_seconds || 300,
        duree: total,
        moment: instant,
        surPhase: (phase) => {
          if (actuelle()) setSonde((s) => (s?.enCours ? { ...s, phase } : s));
        },
      });
      if (!actuelle()) return;
      setSonde({ ...vue, enCours: false, moment: instant });
      const retenu = vue.investigation?.record;
      // Une réunion récupérée a souvent déjà sa transcription dans Odoo,
      // déposée par l'ancienne app : le dépôt ne s'y fait qu'à la main.
      if (retenu) retenir({ ...retenu, auto: Boolean(vue.investigation.auto) && !choisi.remote });
    } catch {
      if (actuelle()) setSonde(null);
    }
  }

  /** Des fichiers arrivent — choisis ou déposés. Le premier s'ouvre dans
   *  le formulaire, les suivants attendent leur tour. Pendant qu'une
   *  transcription part, tous attendent : on ne remplace pas sous les
   *  doigts ce qu'on est en train d'envoyer. */
  function recevoir(liste) {
    const medias = liste.filter(isMedia);
    setIgnores(liste.length - medias.length);
    if (!medias.length) return;
    const occupe = pipeline.etat === 'traitement' || pipeline.etat === 'creation';
    if (occupe) {
      enqueue(medias);
      return;
    }
    charger(medias[0]);
    enqueue(medias.slice(1));
  }

  function choisir(event) {
    const liste = [...(event.target.files || [])];
    // Vidé aussitôt : choisir à nouveau le même fichier doit encore
    // déclencher quelque chose.
    event.target.value = '';
    if (liste.length) recevoir(liste);
  }

  // « Lancer » depuis la bibliothèque : le fichier attendait dans la file.
  useEffect(() => {
    const demande = consumeLaunchRequest();
    if (demande) charger(demande);
  }, []);

  // Quitter l'écran avec un fichier chargé mais pas lancé le remet en tête
  // de la file : il attend dans « À lancer » au lieu de disparaître.
  const enSuspens = useRef(null);
  enSuspens.current = pipeline.etat === 'repos' && !pipeline.jobId ? fichier : null;
  useEffect(() => () => {
    if (enSuspens.current) enqueue([enSuspens.current], { front: true });
  }, []);

  async function charger(original) {
    setFichier(null);
    setSecondes(0);
    sondeCourante.current += 1;
    setSonde(null);
    retirer();
    if (!original) return setLecture('');
    setLecture('Lecture des métadonnées…');
    try {
      const choisi = await copieEnMemoire(original);
      const total = await sonderDuree(choisi);
      setFichier(choisi);
      setSecondes(total);
      setDebut(0);
      setFin(total);
      setLecture(`${mo(choisi.size)} · ${duree(total)}`);
      const datee = choisi.remote?.meetingDate
        ? new Date(choisi.remote.meetingDate.replace(' ', 'T'))
        : null;
      const debutReunion = datee && !Number.isNaN(datee.getTime())
        ? datee
        : new Date((choisi.lastModified || Date.now()) - total * 1000);
      setDateReunion(versChampLocal(debutReunion));
      if (choisi.remote) {
        if (modeChoisi.current === null) modeChoisi.current = mode;
        setMode(choisi.size * 8 > total * DEBIT_A_COMPRESSER ? 'les-deux' : 'transcrire');
        setLecture(`${mo(choisi.size)} · ${duree(total)} · vidéo déjà stockée avec la réunion`);
      } else if (modeChoisi.current !== null) {
        setMode(modeChoisi.current);
        modeChoisi.current = null;
      }
      if (mode !== 'compresser') ecouter(choisi, total, debutReunion.toISOString());
    } catch (e) {
      setFichier(null);
      setLecture(`Fichier illisible : ${e.message}`);
    }
  }

  if (!capacites) {
    return (
      <section className="mx-auto max-w-2xl px-6 py-16">
        <h1 className="titre text-[1.5rem] font-semibold">Navigateur non supporté</h1>
        <p className="mt-3 text-fonce/70">
          Le découpage se fait sur ton poste, ce qui demande WebCodecs et une
          connexion sécurisée. Chrome, Edge ou Safari 26 et plus conviennent.
        </p>
      </section>
    );
  }

  const enCours = pipeline.etat === 'traitement' || pipeline.etat === 'creation';

  /** Verse dans le formulaire ce qu'Odoo vient d'apprendre.
   *
   *  Même geste pour une réunion d'agenda et pour un dossier proposé
   *  par la sonde : ce qui est déjà saisi n'est jamais écrasé, seulement
   *  complété.
   */
  /** Reprend ce que le dossier précédent avait versé, et seulement ça :
   *  une société retapée à la main ou un terme ajouté restent. */
  function reprendreApport() {
    const { client: societe, termes } = apport.current;
    apport.current = { client: '', termes: [] };
    if (societe) setClient((actuel) => (actuel === societe ? '' : actuel));
    if (termes.length) {
      setGlossaire((actuel) => liste(actuel).filter((t) => !termes.includes(t)).join(', '));
    }
    setContexteOdoo('');
  }

  /** Retient un dossier et charge son contexte Odoo. */
  async function retenir(choisi) {
    reprendreApport();
    setDossier(choisi);
    try {
      const pack = await api.odooContext(choisi.model, choisi.id);
      const societe = pack.client_company || '';
      const termes = pack.terms || [];
      setClient((actuel) => {
        if (actuel || !societe) return actuel;
        apport.current.client = societe;
        return societe;
      });
      setGlossaire((actuel) => {
        const deja = liste(actuel);
        const nouveaux = termes.filter((t) => !deja.includes(t));
        apport.current.termes = nouveaux;
        return [...deja, ...nouveaux].join(', ');
      });
      if (pack.summary) setContexteOdoo(pack.summary);
    } catch {
      // Le contexte est un bonus : son échec ne doit pas empêcher de
      // retenir le dossier.
    }
  }

  /** Aucun dossier : ni dépôt à la fin, ni contexte qui en venait. */
  function retirer() {
    setDossier(null);
    reprendreApport();
  }

  function appliquerContexte({ client: societe, termes, resume, invites }) {
    if (societe) setClient((actuel) => actuel || societe);
    if (invites?.length) {
      setParticipants((actuel) =>
        [...new Set([...liste(actuel), ...invites])].join(', '),
      );
    }
    if (termes?.length) {
      setGlossaire((actuel) =>
        [...new Set([...liste(actuel), ...termes])].join(', '),
      );
    }
    if (resume) setContexteOdoo(resume);
  }

  return (
    <section className="mx-auto max-w-3xl px-6 py-10">
      <h1 className="titre text-[1.5rem] font-semibold">Nouvelle transcription</h1>
      <p className="mt-2 max-w-xl text-fonce/70">
        Le découpage et l'encodage se font sur ce poste. Seules des fenêtres
        audio de quelques mégaoctets partent au serveur — ta vidéo, elle, ne
        quitte pas ta machine.
      </p>

      <div className="mt-8 space-y-8">
        <div>
          <h2 className="titre text-[1.0625rem] font-medium">1. Le fichier</h2>
          {/* La zone change d'aspect dès qu'un fichier survole la page,
              et plus franchement quand il la survole elle : on voit où
              lâcher avant d'avoir à viser. */}
          <div
            className="relative mt-3"
            onDragOver={(e) => { e.preventDefault(); setSurZone(true); }}
            onDragLeave={() => setSurZone(false)}
            onDrop={(e) => {
              e.preventDefault();
              setSurZone(false);
              recevoir([...e.dataTransfer.files]);
            }}
          >
            {/* Le champ natif reste, caché : il dirait « Aucun fichier
                choisi » après un dépôt, alors qu'un fichier est chargé. */}
            <input
              id="champ-fichier"
              ref={champFichier}
              type="file"
              multiple
              accept="video/*,audio/*"
              onChange={choisir}
              disabled={enCours}
              className="sr-only"
            />
            <label
              htmlFor="champ-fichier"
              className={`verre flex items-center gap-4 rounded-xl border-dashed px-4 py-5 transition-opacity ${
                enCours ? 'cursor-not-allowed' : 'cursor-pointer hover:bg-white/50'
              } ${glisse ? 'opacity-0' : ''}`}
            >
              <span className={`shrink-0 rounded-md px-3 py-1.5 text-ekn-sm ${
                enCours ? 'bg-fonce/30 text-clair' : 'bg-fonce text-clair'
              }`}>
                Choisir des fichiers
              </span>
              <span className="min-w-0 truncate text-[0.9375rem] text-fonce/65">
                {fichier ? fichier.name : 'ou glisse un ou plusieurs enregistrements ici'}
              </span>
            </label>
            {glisse ? (
              <div
                className={`pointer-events-none absolute inset-0 flex flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all ${
                  surZone
                    ? 'scale-[1.02] border-turquoise-sombre bg-turquoise/25 text-fonce shadow-lg'
                    : 'border-turquoise bg-turquoise/10 text-fonce/75'
                }`}
              >
                <span className="titre text-[0.9375rem] font-medium">
                  {surZone ? 'Lâche pour ajouter' : 'Dépose tes enregistrements ici'}
                </span>
                <span className="text-ekn-sm text-ekn-text-muted">
                  {enCours ? 'ils attendront la fin de l’envoi en cours' : 'un seul, ou plusieurs à la fois'}
                </span>
              </div>
            ) : null}
          </div>
          {ignores ? (
            <p className="mt-2 text-ekn-sm text-ekn-text-muted">
              {ignores} fichier{ignores > 1 ? 's' : ''} ignoré{ignores > 1 ? 's' : ''} : ni audio ni vidéo.
            </p>
          ) : null}
          {enAttente.length ? (
            <div className="mt-2 flex flex-wrap items-center gap-1.5 text-ekn-sm">
              <span className="text-ekn-text-muted">Ensuite :</span>
              {enAttente.map((entree) => (
                <span key={entree.id} className="inline-flex items-center gap-1 rounded-full bg-white/70 px-2.5 py-0.5 text-fonce/75 ring-1 ring-bord">
                  {entree.file.name}
                  <button
                    type="button"
                    onClick={() => removeFromQueue(entree.id)}
                    aria-label={`Retirer ${entree.file.name} de la file`}
                    className="text-ekn-text-muted hover:text-fonce"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          ) : null}
          {lecture ? <p className="mt-2 text-ekn-sm text-fonce/60">{lecture}</p> : null}
          {fichier ? (
            <div className="mt-3 flex flex-wrap items-center gap-3 text-ekn-sm">
              <span className="text-fonce/70">Date de la réunion</span>
              <DateHeure valeur={dateReunion} surChange={setDateReunion} disabled={enCours} />
              <span className="text-ekn-sm text-ekn-text-muted">
                {fichier.remote
                  ? 'celle de la réunion récupérée'
                  : "déduite du fichier — corrige-la s'il a été recopié"}
              </span>
            </div>
          ) : null}
          <Apercu
            fichier={fichier}
            duree={secondes}
            debut={debut}
            fin={fin || secondes}
            surDebut={setDebut}
            surFin={setFin}
            actif={enCours}
          />
        </div>

        <div>
          <h2 className="titre text-[1.0625rem] font-medium">2. Que faire de ce fichier</h2>
          <div className="mt-3 flex flex-wrap gap-2">
            {[
              ['transcrire', 'Transcrire'],
              // La vidéo d'une réunion retraitée est déjà stockée : la
              // compresser n'a de sens que pour l'y remplacer.
              ...(fichier?.remote ? [] : [['compresser', 'Compresser']]),
              ['les-deux', fichier?.remote ? 'Transcrire et alléger la vidéo' : 'Les deux'],
            ].map(([cle, libelle]) => (
              <button
                key={cle}
                type="button"
                onClick={() => setMode(cle)}
                aria-pressed={mode === cle}
                className={`titre rounded-lg px-3 py-1.5 text-[0.9375rem] font-medium transition-colors ${
                  mode === cle
                    ? 'bg-fonce text-clair'
                    : 'border border-bord bg-white hover:border-fonce/40'
                }`}
              >
                {libelle}
              </button>
            ))}
          </div>
          {mode === 'les-deux' && fichier?.remote ? (
            <p className="mt-2 text-ekn-sm text-ekn-text-muted">
              La vidéo stockée ({mo(fichier.size)}) est relue, compressée sur ce
              poste — 720p, HEVC — puis remplace l’originale en stockage froid.
              Si elle n’en sort pas plus légère, on garde celle qui y est.
            </p>
          ) : mode === 'les-deux' && videoDisponible ? (
            <p className="mt-2 text-ekn-sm text-ekn-text-muted">
              La vidéo est compressée sur ce poste — 720p, HEVC, environ 92 %
              plus légère — puis archivée en stockage froid avec la réunion.
              Ton fichier d'origine ne quitte pas ta machine.
            </p>
          ) : mode !== 'transcrire' ? (
            <p className="mt-2 text-ekn-sm text-ekn-text-muted">
              La version compressée est enregistrée sur ton disque — 720p,
              HEVC, environ 92 % plus légère.
              {mode === 'compresser' && videoDisponible
                ? ' Pour l’archiver avec la réunion, choisis « Les deux ».'
                : ''}
              {!compression.supportee
                ? " Ce navigateur ne sait pas écrire un fichier sur le disque : utilise Chrome ou Edge."
                : ''}
            </p>
          ) : null}
        </div>

        <div className={mode === 'compresser' ? 'hidden' : undefined}>
          <h2 className="titre text-[1.0625rem] font-medium">3. Le contexte</h2>
          <p className="mt-1 text-ekn-sm text-fonce/60">
            Facultatif, mais c'est ce qui fait la différence entre « Réunion du
            3 juillet » et un titre utile.
          </p>
          <Sonde
            // Une réunion, une conversation : la recherche guidée d'un
            // fichier ne doit pas suivre le suivant, qui est un autre dossier.
            key={fichier ? `${fichier.name}|${fichier.size}|${fichier.lastModified}` : 'aucun'}
            etat={sonde}
            retenu={dossier}
            // Choisir un dossier, c'est le valider : la transcription y
            // sera déposée à la fin, sauf à décocher la case plus bas.
            surChoix={(choisi) => retenir({ ...choisi, auto: !fichier?.remote })}
            surRetrait={retirer}
            surAjout={(trouve) => {
              // Trouvé en guidant la recherche : il rejoint la liste,
              // retenu, pour qu'on voie d'où il vient et qu'on puisse
              // encore revenir aux autres.
              const ajoute = { ...trouve, reason: trouve.reason || 'trouvé en guidant la recherche' };
              setSonde((s) => ({
                ...s,
                candidates: [
                  ...(s?.candidates || []).filter(
                    (d) => !(d.id === ajoute.id && d.model === ajoute.model),
                  ),
                  ajoute,
                ],
              }));
              retenir({ ...ajoute, auto: !fichier?.remote });
            }}
          />
          <Reunions
            surChoix={(choix) => {
              appliquerContexte(choix);
              // La réunion d'agenda pointe souvent déjà un dossier : le
              // choisir doit le retenir, sinon la liaison se perdait et
              // rien n'était déposé à la fin.
              if (choix.dossier) setDossier({ ...choix.dossier, auto: !fichier?.remote });
            }}
          />
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <Champ
              label="Partie prenante"
              aide="Préfixe du titre : « Acritec - Sujet »"
              placeholder="Acritec"
              value={client}
              onChange={(e) => setClient(e.target.value)}
              disabled={enCours}
            />
            <Champ
              label="Participants attendus"
              aide="Séparés par des virgules"
              placeholder="Robin, Lùka"
              value={participants}
              onChange={(e) => setParticipants(e.target.value)}
              disabled={enCours}
            />
          </div>
          <div className="mt-4">
            <Champ
              label="Vocabulaire métier"
              aide="Les termes que le modèle risque d'écorcher"
              placeholder="Odoo, Wedophone, EDOF"
              value={glossaire}
              onChange={(e) => setGlossaire(e.target.value)}
              disabled={enCours}
            />
            <Suggestions
              choisis={[...liste(glossaire), client.trim()].filter(Boolean)}
              surAjout={(terme) =>
                setGlossaire((actuel) => (actuel.trim() ? `${actuel}, ${terme}` : terme))
              }
            />
          </div>
        </div>

        {dossier && mode !== 'compresser' ? (
          <div className="rounded-xl border border-turquoise/40 bg-white/80 px-4 py-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-ekn-sm">
                <span className="text-ekn-text-muted">Dossier Odoo : </span>
                <span className="titre font-medium">{dossier.name}</span>
                {dossier.partner ? <span className="text-ekn-text-muted"> · {dossier.partner}</span> : null}
              </p>
              <button
                type="button"
                disabled={enCours}
                onClick={retirer}
                className="text-ekn-sm text-ekn-text-muted hover:text-fonce"
              >
                retirer
              </button>
            </div>
            <label className="mt-2 flex items-center gap-2 text-ekn-sm text-fonce/80">
              <input
                type="checkbox"
                checked={Boolean(dossier.auto)}
                disabled={enCours}
                onChange={(e) => setDossier({ ...dossier, auto: e.target.checked })}
                className="h-4 w-4 accent-turquoise"
              />
              Déposer la transcription dans ce dossier à la fin, et me prévenir
            </label>
            {fichier?.remote ? (
              <p className="mt-1 pl-6 text-ekn-sm text-ekn-text-muted">
                Réunion récupérée : sa transcription y est peut-être déjà,
                déposée par l’ancienne app. Coche seulement si ce n’est pas le cas.
              </p>
            ) : null}
          </div>
        ) : null}

        <div className="flex flex-wrap items-center gap-4">
          <Bouton
            variante="accent"
            disabled={!fichier || enCours || (mode === 'compresser' && !compression.supportee)}
            onClick={() => {
              const borne = debut > 0 || (fin && fin < secondes)
                ? { start: debut, end: fin || secondes }
                : null;
              if (mode === 'les-deux' && videoDisponible) {
                // Retraitée, la réunion existe déjà : l'envoi n'a pas à
                // attendre sa création.
                const cible = { cle: `nouveau-${Date.now()}`, jobId: fichier.remote?.jobId ?? null };
                cibleRef.current = cible;
                archiver({ fichier, trim: borne, cible });
              } else if (mode !== 'transcrire') {
                compression.compresser(fichier, borne);
              }
              if (mode === 'compresser') return;
              pipeline.lancer({
                meetingDate: dateReunion ? new Date(dateReunion).toISOString() : null,
                fichier,
                // Seule la partie retenue est transcrite — donc payée.
                duree: (fin || secondes) - debut,
                offset: debut,
                modele: MODELE,
                contexte: {
                  client_company: client.trim(),
                  expected_speaker_names: liste(participants),
                  glossary_terms: liste(glossaire),
                  odoo_context: contexteOdoo,
                  // Décidé avant de partir : à la fin, le dépôt n'aura
                  // plus rien à demander.
                  ...(dossier
                    ? {
                        odoo_record: { model: dossier.model, record_id: dossier.id },
                        odoo_auto: dossier.auto,
                      }
                    : {}),
                },
              });
            }}
          >
            {enCours
              ? 'Traitement en cours…'
              : mode === 'compresser'
                ? 'Compresser'
                : mode === 'les-deux'
                  ? 'Compresser et transcrire'
                  : 'Lancer la transcription'}
          </Bouton>
          {pipeline.estimation !== null ? (
            <span className="text-ekn-sm text-fonce/60">
              Coût estimé {usd(pipeline.estimation)}
            </span>
          ) : null}
        </div>

        <Erreur>{pipeline.erreur}</Erreur>
        {pipeline.reprenable ? (
          // Le fichier est encore en mémoire et la réunion existe : on
          // reprend où ça s'est arrêté, sans repayer ce qui est transcrit.
          <div className="-mt-4 flex flex-wrap items-center gap-3 text-ekn-sm">
            <Bouton onClick={() => reprendre(pipeline.cle)}>Reprendre là où ça s’est arrêté</Bouton>
            <span className="text-ekn-text-muted">
              Si le fichier est sur iCloud, attends qu’il soit entièrement téléchargé sur ce Mac.
            </span>
          </div>
        ) : null}
        <Erreur>{compression.erreur}</Erreur>
        {archivage ? <AvancementArchivage tache={archivage} /> : null}
        <Compression compression={compression} />

        {pipeline.fenetres.length > 0 ? (
          <Avancement
            numero={mode === 'compresser' ? 3 : 4}
            fenetres={pipeline.fenetres}
            message={pipeline.message}
          />
        ) : null}

        {pipeline.etat === 'traitement' || pipeline.etat === 'interrompu' ? (
          <div className="verre rounded-xl p-4">
            <p className="text-ekn-sm text-fonce/75">
              {pipeline.etat === 'interrompu'
                ? 'Transcription interrompue. La réunion est dans la corbeille, si tu veux la récupérer.'
                : pipeline.envoiTermine
                  ? 'Tout est envoyé : le serveur termine seul. Tu peux fermer cet onglet.'
                  : 'Tu peux faire autre chose pendant ce temps — garde seulement cet onglet ouvert tant que l’envoi n’est pas fini.'}
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <Bouton onClick={() => { pipeline.detacher(); surBibliotheque?.(); }}>
                Revenir à la bibliothèque
              </Bouton>
              <button
                type="button"
                onClick={() => {
                  // La transcription en cours continue ; on repart d'un
                  // formulaire vierge pour la suivante — et s'il y a des
                  // fichiers en attente, avec le prochain déjà chargé.
                  pipeline.detacher();
                  setFichier(null); setSecondes(0); setLecture(''); setSonde(null);
                  setDossier(null); setClient(''); setParticipants(''); setGlossaire('');
                  setContexteOdoo(''); setDateReunion(''); setDebut(0); setFin(0);
                  cibleRef.current = null;
                  sondeCourante.current += 1;
                  apport.current = { client: '', termes: [] };
                  if (champFichier.current) champFichier.current.value = '';
                  const suivant = takeNext();
                  if (suivant) charger(suivant);
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className="ekn-button ekn-button--subtle"
              >
                {enAttente.length
                  ? `Passer à la suivante (${enAttente.length} en attente)`
                  : 'Lancer une autre transcription'}
              </button>
              {pipeline.etat === 'traitement' ? (
                <button
                  type="button"
                  onClick={() => {
                    if (window.confirm(
                      'Interrompre cette transcription ? Ce qui n’est pas encore transcrit ne le sera pas, '
                      + 'et la réunion part à la corbeille.',
                    )) interrompre(pipeline.cle);
                  }}
                  className="ekn-button ekn-button--danger ml-auto"
                >
                  Interrompre
                </button>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}

const LIBELLES = {
  attendue: 'en attente',
  encodage: 'encodage sur ce poste',
  envoi: 'envoi',
  transcription: 'transcription',
  terminee: 'terminée',
  erreur: 'erreur',
};

/** Avancement par fenêtre.
 *
 *  Une ligne par fenêtre plutôt qu'une barre unique : sur une réunion de
 *  trois heures, savoir *laquelle* patine est la seule information qui
 *  aide — c'est aussi elle qu'on pourra relancer seule.
 */
function Avancement({ numero, fenetres, message }) {
  const finies = fenetres.filter((f) => f.etat === 'terminee').length;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        {/* Le contexte est masqué en simple compression : la numérotation suit
            ce qui est affiché, sans doublon ni trou. */}
        <h2 className="titre text-[1.0625rem] font-medium">{numero}. Avancement</h2>
        <span className="text-ekn-sm text-fonce/60">
          {finies} / {fenetres.length} fenêtres
        </span>
      </div>

      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-fonce/10">
        <div
          className="h-full rounded-full bg-turquoise transition-[width] duration-500"
          style={{ width: `${(finies / fenetres.length) * 100}%` }}
        />
      </div>
      {message ? <p className="mt-2 text-ekn-sm text-fonce/60">{message}</p> : null}

      <ul className="verre mt-4 divide-y divide-bord/60 rounded-xl">
        {fenetres.map((f) => (
          <li key={f.index} className="flex items-center gap-4 px-4 py-2.5">
            <span className="w-16 shrink-0 text-ekn-sm tabular-nums text-ekn-text-muted">
              {horodatage(f.start)}
            </span>
            <span className={`flex-1 text-[0.9375rem] ${f.etat === 'erreur' ? 'text-ekn-error-dark' : ''}`}>
              {f.etat === 'erreur' ? f.erreur : LIBELLES[f.etat]}
              {f.etat === 'encodage' && f.progression
                ? ` ${Math.round(f.progression * 100)} %`
                : ''}
            </span>
            <span className="shrink-0 text-ekn-sm tabular-nums text-ekn-text-muted">
              {f.octets ? mo(f.octets) : ''}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Vocabulaire déjà connu de l'équipe.
 *
 *  Trié par affinité avec ce qui est déjà saisi, pas par fréquence :
 *  saisir « Acritec » doit faire remonter les termes de ce client, et
 *  non les cinq mêmes mots présents dans toutes les réunions.
 */
function Suggestions({ choisis, surAjout }) {
  const [termes, setTermes] = useState([]);
  const cle = choisis.join(',');

  useEffect(() => {
    let vivant = true;
    const attente = setTimeout(() => {
      api.vocabulary(choisis)
        .then((v) => vivant && setTermes(v.slice(0, 12)))
        .catch(() => {});
    }, 200);
    return () => { vivant = false; clearTimeout(attente); };
  }, [cle]);

  if (termes.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5">
      <span className="text-ekn-sm text-ekn-text-muted">Déjà utilisés :</span>
      {termes.map((t) => (
        <button
          key={t.term}
          type="button"
          onClick={() => surAjout(t.term)}
          className="rounded-md border border-bord bg-white px-2 py-0.5 text-ekn-sm hover:border-ekn-border-control"
        >
          {t.term}
        </button>
      ))}
    </div>
  );
}

/** Réunions Odoo du moment.
 *
 *  Odoo enrichit, il ne conditionne pas : s'il est absent ou en panne,
 *  ce bloc disparaît simplement. Rien n'empêche de transcrire.
 */
function Reunions({ surChoix }) {
  const [etat, setEtat] = useState(null);
  const [choisie, setChoisie] = useState(null);
  const [chargement, setChargement] = useState(false);

  useEffect(() => {
    api.odooMeetings().then(setEtat).catch(() => setEtat({ available: false, meetings: [] }));
  }, []);

  if (!etat?.available || etat.meetings.length === 0) return null;

  return (
    <div className="verre mt-4 rounded-xl p-4">
      <p className="titre text-[0.9375rem] font-medium">Réunions Odoo du moment</p>
      <p className="mt-0.5 text-ekn-sm text-ekn-text-muted">
        En choisir une remplit les participants, la partie prenante et le
        vocabulaire depuis Odoo.
      </p>
      <ul className="mt-3 space-y-1">
        {etat.meetings.map((r) => (
          <li key={r.id}>
            <button
              type="button"
              disabled={chargement}
              onClick={async () => {
                setChoisie(r.id);
                setChargement(true);
                try {
                  const pack = r.resource_model
                    ? await api.odooContext(r.resource_model, r.resource_id)
                    : {};
                  surChoix({
                    client: pack.client_company,
                    termes: pack.terms,
                    resume: pack.summary,
                    // Les invités valent même sans fiche liée : c'est le
                    // cas le plus fréquent, et savoir qui parle change
                    // l'attribution des répliques.
                    invites: r.attendees,
                    dossier: r.resource_model && r.resource_id
                      ? { model: r.resource_model, id: r.resource_id,
                          name: `rattaché à « ${r.name} »`, kind: 'réunion d’agenda' }
                      : null,
                  });
                } catch {
                  // Le contexte est un bonus : son échec ne doit pas
                  // empêcher de retenir la réunion choisie.
                } finally {
                  setChargement(false);
                }
              }}
              className={`w-full rounded-md px-2 py-1.5 text-left text-ekn-sm hover:bg-papier ${
                choisie === r.id ? 'ring-1 ring-turquoise-sombre' : ''
              }`}
            >
              <span className="titre font-medium">{r.name}</span>
              {r.attendees.length ? (
                <span className="block text-ekn-sm text-ekn-text-muted">
                  {r.attendees.join(', ')}
                </span>
              ) : null}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Où en est l'archivage de la vidéo, tant qu'on est sur cet écran.
 *  La fiche de la réunion prend le relais une fois qu'on l'a ouverte. */
export function AvancementArchivage({ tache }) {
  const libelle = {
    compression: 'Compression sur ce poste',
    envoi: 'Envoi vers le stockage froid',
    termine: tache.note ? 'Vidéo conservée' : 'Vidéo archivée',
    erreur: 'Archivage interrompu',
  }[tache.etape] || 'Archivage';
  return (
    <div className="mt-2">
      <p className="text-ekn-sm text-fonce/70">
        {libelle}
        {tache.etape === 'compression' || tache.etape === 'envoi'
          ? ` — ${Math.round((tache.progression || 0) * 100)} %. Tu peux continuer à travailler, mais garde cet onglet ouvert.`
          : ''}
      </p>
      {tache.etape === 'compression' || tache.etape === 'envoi' ? (
        <div className="mt-1 h-1.5 w-full max-w-md overflow-hidden rounded-full bg-bord">
          <div
            className="h-full bg-turquoise transition-[width]"
            style={{ width: `${(tache.progression || 0) * 100}%` }}
          />
        </div>
      ) : null}
      {tache.note ? <p className="mt-1 text-ekn-sm text-fonce/60">{tache.note}</p> : null}
      {tache.erreur ? <p className="mt-1 text-ekn-sm text-violet">{tache.erreur}</p> : null}
    </div>
  );
}

/** Avancement de la compression.
 *
 *  Séparé de la transcription parce que les deux avancent en parallèle
 *  et à des rythmes très différents : l'audio part en quelques minutes,
 *  la vidéo prend des heures sur une longue réunion.
 */
function Compression({ compression }) {
  if (compression.etat === 'repos' && !compression.resultat) return null;
  const { resultat } = compression;
  return (
    <div>
      <h2 className="titre text-[1.0625rem] font-medium">Compression</h2>
      {resultat ? (
        <p className="mt-2 text-[0.9375rem]">
          {mo(resultat.source)} → <strong>{mo(resultat.bytes)}</strong>{' '}
          <span className="text-ekn-text-muted">
            ({Math.round((1 - resultat.bytes / resultat.source) * 100)} % de moins,
            en {Math.round(resultat.ms / 60000)} min) — enregistré sur ton disque.
          </span>
        </p>
      ) : (
        <>
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-fonce/10">
            <div
              className="h-full rounded-full bg-turquoise transition-[width] duration-500"
              style={{ width: `${compression.progression * 100}%` }}
            />
          </div>
          <p className="mt-2 text-ekn-sm text-fonce/60">
            Encodage sur ce poste — {Math.round(compression.progression * 100)} %.
            Garde cet onglet ouvert.
          </p>
        </>
      )}
    </div>
  );
}
