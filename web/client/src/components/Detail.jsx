import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api.js';
import { Bouton, Champ, DateHeure, Erreur } from './Communs.jsx';
import { horodatage, usd, jour, mo } from '../format.js';
import { useArchivage } from '../archivage.js';
import { AvancementArchivage } from './Nouveau.jsx';
import { MOD, useRaccourcis } from '../raccourcis.js';
import { MediaPlayer } from './MediaPlayer.jsx';
import { plier, surligner } from '../surlignage.jsx';
import { launch, remoteFile } from '../fileQueue.js';

/** Fiche d'une transcription : la lire, la corriger, la relancer.
 *
 *  Les éditeurs sont posés à côté du transcript plutôt que dans une
 *  fenêtre modale : corriger un nom d'interlocuteur se fait en le
 *  lisant, pas de mémoire.
 */
export function Detail({ jobId, recherche = '', surRetour, surRetraiter }) {
  const [fiche, setFiche] = useState(null);
  const [erreur, setErreur] = useState('');
  const [note, setNote] = useState('');
  const [surlignee, setSurlignee] = useState(null);
  // Pendant l'écoute : l'instant entendu, pour suivre la réplique en cours
  // dans la transcription, et le lecteur, pour y sauter depuis un horodatage.
  const [ecoute, setEcoute] = useState(null);
  const lecteur = useRef(null);
  // Le mot douteux reste marqué dans sa réplique jusqu'au clic suivant :
  // le temps de le lire, et de le corriger.
  const [marque, setMarque] = useState(null);
  const [aCorriger, setACorriger] = useState(null);

  /** Amène la transcription au mot douteux.
   *
   *  L'horodatage du modèle est approximatif — jusqu'à quarante secondes
   *  d'écart constatées. Le mot lui-même est le vrai repère : on prend
   *  la réplique qui le contient au plus près de l'instant annoncé, et
   *  l'instant seul ne sert qu'à défaut.
   */
  const allerA = (secondes, terme = '') => {
    const segments = fiche?.segments || [];
    if (!segments.length) return;
    let rang = -1;
    if (terme) {
      let ecart = Infinity;
      segments.forEach((seg, i) => {
        if (!plier(seg.text).includes(plier(terme))) return;
        const d = secondes == null ? 0 : Math.abs(seg.start_second - secondes);
        if (d < ecart) { ecart = d; rang = i; }
      });
    }
    if (rang < 0) {
      if (secondes == null) return;
      rang = 0;
      segments.forEach((seg, i) => { if (seg.start_second <= secondes) rang = i; });
    }
    document.getElementById(`segment-${rang}`)
      ?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    setTrouver(null);
    setMarque(terme ? { rang, terme } : null);
    setSurlignee(rang);
    setTimeout(() => setSurlignee((r) => (r === rang ? null : r)), 2500);
  };

  // Chercher dans la transcription : ouverte d'office quand on arrive
  // d'une recherche de la bibliothèque, pour voir tout de suite où.
  const [trouver, setTrouver] = useState(recherche ? { terme: recherche, courant: 0 } : null);
  const champTrouver = useRef(null);
  const occurrences = useMemo(
    () => (trouver?.terme ? occurrencesDe(fiche?.segments || [], trouver.terme) : []),
    [fiche, trouver?.terme],
  );
  const actuelle = occurrences.length ? occurrences[trouver.courant % occurrences.length] : null;

  useEffect(() => {
    if (!actuelle) return;
    document.getElementById(`segment-${actuelle.rang}`)
      ?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }, [actuelle?.rang, actuelle?.k]);

  const ouvrirTrouver = () => {
    setTrouver((t) => t || { terme: '', courant: 0 });
    setTimeout(() => { champTrouver.current?.focus(); champTrouver.current?.select(); }, 0);
  };
  const avancer = (pas) => setTrouver((t) => (t && occurrences.length
    ? { ...t, courant: (t.courant + pas + occurrences.length) % occurrences.length }
    : t));

  useRaccourcis({
    'mod+f': () => { if (!fiche?.segments?.length) return false; ouvrirTrouver(); },
    'mod+g': () => (trouver ? avancer(1) : false),
    'shift+mod+g': () => (trouver ? avancer(-1) : false),
    escape: (e) => {
      // Une fenêtre ouverte (aide, confirmation) se ferme d'abord.
      if (document.querySelector('[role="dialog"]')) return false;
      if (trouver) { setTrouver(null); return undefined; }
      // Dans un champ, Échap rend la main sans quitter : on ne perd pas
      // une correction en cours sur une touche.
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) { e.target.blur(); return undefined; }
      surRetour();
      return undefined;
    },
  });

  const recharger = () => api.detail(jobId).then(setFiche).catch((e) => setErreur(e.message));
  useEffect(() => { recharger(); }, [jobId]);

  if (erreur && !fiche) return <div className="mx-auto max-w-4xl px-6 py-10"><Erreur>{erreur}</Erreur></div>;
  if (!fiche) return <p className="py-16 text-center text-fonce/50">Chargement…</p>;

  return (
    <section className="mx-auto max-w-6xl px-6 py-10">
      <button onClick={surRetour} className="text-[0.875rem] text-fonce/55 hover:text-fonce">
        ← Bibliothèque
      </button>

      <h1 className="titre mt-3 text-[1.5rem] font-semibold">{fiche.title || fiche.filename}</h1>
      <p className="mt-1 text-[0.875rem] text-fonce/55">
        {fiche.filename} · {fiche.model} · {usd(fiche.cost_usd)}
      </p>
      <DateReunion
        jobId={jobId}
        valeur={fiche.meeting_date || fiche.created_at}
        deduite={!fiche.meeting_date}
        surMaj={recharger}
        surErreur={setErreur}
      />

      <Erreur>{erreur}</Erreur>
      {note ? <p className="mt-4 text-[0.875rem] text-turquoise-sombre">{note}</p> : null}

      <div className="mt-8 grid gap-10 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div>
          <div className="flex items-center justify-between gap-3">
            <h2 className="titre text-[1.0625rem] font-medium">Transcription</h2>
            <div className="flex flex-wrap gap-2">
              {fiche.segments.length ? (
                <button
                  type="button"
                  onClick={ouvrirTrouver}
                  title={`Chercher dans la transcription (${MOD} F)`}
                  className="rounded-md px-3 py-1.5 text-[0.8125rem] text-fonce/70 ring-1 ring-bord transition-colors hover:bg-white/60 hover:text-fonce"
                >
                  Chercher
                </button>
              ) : null}
              <Relecture fiche={fiche} jobId={jobId} surMaj={recharger} surNote={setNote} surErreur={setErreur} />
              <Copier texte={fiche.transcript} />
            </div>
          </div>
          <Video jobId={jobId} video={fiche.video} filename={fiche.filename} surMaj={recharger}
                 lecteur={lecteur} surTemps={setEcoute} />
          {trouver ? (
            <div className="mt-3 flex items-center gap-2 rounded-lg border border-bord bg-white px-3 py-1.5">
              <input
                ref={champTrouver}
                autoFocus={!recherche}
                value={trouver.terme}
                onChange={(e) => setTrouver({ terme: e.target.value, courant: 0 })}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') { e.preventDefault(); avancer(e.shiftKey ? -1 : 1); }
                }}
                placeholder="Chercher dans la transcription"
                aria-label="Chercher dans la transcription"
                className="min-w-0 flex-1 bg-transparent text-[0.875rem] outline-none placeholder:text-fonce/35"
              />
              <span className="shrink-0 text-[0.8125rem] tabular-nums text-fonce/50">
                {trouver.terme.trim()
                  ? occurrences.length ? `${(trouver.courant % occurrences.length) + 1} / ${occurrences.length}` : 'aucune'
                  : ''}
              </span>
              <button type="button" onClick={() => avancer(-1)} disabled={!occurrences.length}
                      aria-label="Occurrence précédente" title="Précédente (Maj+Entrée)"
                      className="rounded px-1.5 text-fonce/55 hover:bg-papier hover:text-fonce disabled:opacity-30">↑</button>
              <button type="button" onClick={() => avancer(1)} disabled={!occurrences.length}
                      aria-label="Occurrence suivante" title="Suivante (Entrée)"
                      className="rounded px-1.5 text-fonce/55 hover:bg-papier hover:text-fonce disabled:opacity-30">↓</button>
              <button type="button" onClick={() => setTrouver(null)} aria-label="Fermer la recherche"
                      title="Fermer (Échap)"
                      className="rounded px-1.5 text-[1.125rem] leading-none text-fonce/45 hover:bg-papier hover:text-fonce">×</button>
            </div>
          ) : null}
          <div className="verre mt-3 max-h-[34rem] overflow-y-auto rounded-xl">
            {fiche.segments.length === 0 ? (
              fiche.reprocessable && fiche.video?.presente ? (
                <Retraiter fiche={fiche} jobId={jobId} surRetraiter={surRetraiter} />
              ) : (
                <p className="px-4 py-6 text-fonce/50">Pas encore de segment.</p>
              )
            ) : (
              <ol>
                {fiche.segments.map((s, rang) => (
                  <li
                    key={rang}
                    id={`segment-${rang}`}
                    className={`flex gap-4 px-4 py-2 transition-colors duration-700 ${
                      surlignee === rang
                        ? 'bg-turquoise/25'
                        : enCours(fiche.segments, rang, ecoute) ? 'bg-turquoise/10 shadow-[inset_3px_0_0_#2AD39F]' : ''
                    }`}
                  >
                    {ecoute !== null ? (
                      // Pendant l'écoute, l'horodatage mène à cet instant.
                      <button
                        type="button"
                        onClick={() => lecteur.current?.seek(s.start_second)}
                        title="Écouter à partir d’ici"
                        className="w-12 shrink-0 pt-0.5 text-left text-[0.8125rem] tabular-nums text-turquoise-sombre hover:underline"
                      >
                        {horodatage(s.start_second)}
                      </button>
                    ) : (
                      <span className="w-12 shrink-0 pt-0.5 text-[0.8125rem] tabular-nums text-fonce/40">
                        {horodatage(s.start_second)}
                      </span>
                    )}
                    <span>
                      {s.speaker ? (
                        <span className="titre mr-2 font-medium text-turquoise-sombre">{s.speaker}</span>
                      ) : null}
                      {trouver?.terme.trim()
                        ? surligner(s.text, trouver.terme, actuelle?.rang === rang ? actuelle.k : -1)
                        : marque?.rang === rang ? surligner(s.text, marque.terme) : s.text}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </div>
          <Versions
            versions={fiche.previous_versions}
            jobId={jobId}
            surMaj={recharger}
            surNote={setNote}
            surErreur={setErreur}
          />
        </div>

        <aside className="space-y-8">
          <Interlocuteurs fiche={fiche} jobId={jobId} surMaj={recharger} surNote={setNote} surErreur={setErreur} />
          <Termes fiche={fiche} jobId={jobId} surMaj={recharger} surNote={setNote} surErreur={setErreur}
                  aCorriger={aCorriger} />
          <AVerifier
            fiche={fiche}
            surAller={allerA}
            surCorriger={(mot) => setACorriger({ mot, fois: Date.now() })}
          />
          <Fenetres jobId={jobId} surNote={setNote} surErreur={setErreur} />
          <Odoo fiche={fiche} jobId={jobId} surMaj={recharger} surNote={setNote} surErreur={setErreur} />
        </aside>
      </div>
    </section>
  );
}

/** Transcrire une réunion qui n'a que sa vidéo — récupérée, ou dont la
 *  transcription a échoué. La vidéo stockée est lue à distance : rien à
 *  retrouver sur le poste. */
function Retraiter({ fiche, jobId, surRetraiter }) {
  const lancer = () => {
    launch(remoteFile({
      job_id: jobId,
      filename: fiche.filename,
      video_bytes: fiche.video.octets,
      meeting_date: fiche.meeting_date,
    }));
    surRetraiter?.();
  };
  return (
    <div className="px-4 py-6">
      <p className="text-fonce/70">
        {{
          recovered: 'Réunion récupérée : la vidéo est là, mais pas encore de transcription.',
          cancelled: 'Transcription interrompue : la vidéo est toujours stockée.',
        }[fiche.status] || 'La transcription a échoué, mais la vidéo est stockée.'}
      </p>
      <p className="mt-1 text-[0.8125rem] text-fonce/55">
        Le traitement relit la vidéo stockée, sans rien déposer dans Odoo
        d'office ; une vidéo lourde peut être allégée au passage.
      </p>
      <Bouton className="mt-3" onClick={lancer}>Retraiter</Bouton>
    </div>
  );
}

/** La date de la réunion — celle où elle a eu lieu, pas celle du dépôt.
 *  Corrigeable sur place : c'est elle qui trie la bibliothèque. */
function DateReunion({ jobId, valeur, deduite, surMaj, surErreur }) {
  const [edition, setEdition] = useState(false);
  const [saisie, setSaisie] = useState('');
  const date = valeur ? new Date(valeur.replace(' ', 'T')) : null;
  const lisible = date && !Number.isNaN(date.getTime())
    ? date.toLocaleString('fr-FR', { dateStyle: 'full', timeStyle: 'short' })
    : 'date inconnue';

  if (!edition) {
    return (
      <p className="mt-1 text-[0.875rem] text-fonce/70">
        Réunion du {lisible}
        {deduite ? <span className="text-fonce/45"> (date du dépôt)</span> : null}
        <button
          type="button"
          onClick={() => {
            const d = date && !Number.isNaN(date.getTime()) ? date : new Date();
            const local = new Date(d.getTime() - d.getTimezoneOffset() * 60000);
            setSaisie(local.toISOString().slice(0, 16));
            setEdition(true);
          }}
          className="ml-2 text-[0.8125rem] text-turquoise-sombre underline-offset-2 hover:underline"
        >
          modifier
        </button>
      </p>
    );
  }
  return (
    <form
      className="mt-1 flex flex-wrap items-center gap-2 text-[0.875rem]"
      onSubmit={async (e) => {
        e.preventDefault();
        try {
          await api.patch(jobId, { meeting_date: new Date(saisie).toISOString() });
          setEdition(false);
          surMaj();
        } catch (erreur) { surErreur(erreur.message); }
      }}
    >
      <DateHeure valeur={saisie} surChange={setSaisie} />
      <Bouton type="submit">Enregistrer</Bouton>
      <button type="button" onClick={() => setEdition(false)}
              className="text-[0.8125rem] text-fonce/55 hover:text-fonce">
        Annuler
      </button>
    </form>
  );
}

/** La vidéo de la réunion, en stockage froid.
 *
 *  Le stockage froid se paie à l'envers de l'intuition : garder une
 *  vidéo ne coûte presque rien, la relire coûte davantage. On prend
 *  l'habitude dès maintenant, avant que GCS ne rende ce coût réel — d'où
 *  une question avant chaque lecture, et rien qui se charge tout seul.
 */
/** La réplique qu'on entend : la dernière commencée à cet instant. */
function enCours(segments, rang, instant) {
  if (instant === null) return false;
  const debut = segments[rang].start_second;
  const suivante = segments[rang + 1]?.start_second ?? Infinity;
  return instant >= debut && instant < suivante;
}

const AUDIO = /\.(m4a|mp3|wav|aac|flac|ogg|opus|amr|weba)$/i;

function Video({ jobId, video, filename, surMaj, lecteur, surTemps }) {
  const audio = AUDIO.test(filename || '');
  const tache = useArchivage(jobId);
  const [etape, setEtape] = useState('repos'); // repos → question → lecture

  useEffect(() => {
    if (tache?.etape === 'termine') surMaj();
  }, [tache?.etape]);

  if (tache && tache.etape !== 'termine') {
    return <div className="mt-3"><AvancementArchivage tache={tache} /></div>;
  }
  if (!video?.presente) {
    return video?.en_cours ? (
      <p className="mt-3 text-[0.8125rem] text-fonce/55">
        L'envoi de la vidéo a été interrompu (onglet fermé ?). La transcription,
        elle, est complète.
      </p>
    ) : null;
  }

  if (etape === 'lecture') {
    return (
      <MediaPlayer
        ref={lecteur}
        src={`/api/jobs/${jobId}/video`}
        autoPlay
        onTime={surTemps}
      />
    );
  }

  return (
    <div className="verre mt-3 rounded-xl p-4">
      {etape === 'question' ? (
        <>
          <p className="titre text-[0.9375rem] font-medium">{audio ? 'Réécouter l’enregistrement ?' : 'Relire la vidéo ?'}</p>
          <p className="mt-1 text-[0.875rem] text-fonce/70">
            {audio ? 'Il' : 'Elle'} est en <strong>stockage froid</strong> : {audio ? 'le' : 'la'} conserver
            ne coûte presque rien, {audio ? 'le réécouter' : 'la relire'} coûte davantage. On ne{' '}
            {audio ? 'le' : 'la'} charge que si tu en as besoin.
          </p>
          <div className="mt-3 flex gap-2">
            <Bouton onClick={() => setEtape('lecture')}>{audio ? 'Écouter' : 'Lire la vidéo'}</Bouton>
            <button
              type="button"
              onClick={() => setEtape('repos')}
              className="rounded-md px-3 py-1.5 text-[0.875rem] text-fonce/60 hover:text-fonce"
            >
              Pas maintenant
            </button>
          </div>
        </>
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-[0.875rem] text-fonce/70">
            {audio ? 'Enregistrement archivé' : 'Vidéo archivée'}{video.octets ? ` — ${mo(video.octets)}` : ''}, en
            stockage froid.
          </p>
          <button
            type="button"
            onClick={() => setEtape('question')}
            className="rounded-md px-3 py-1.5 text-[0.8125rem] text-fonce/70 ring-1 ring-bord hover:bg-white/60 hover:text-fonce"
          >
            {audio ? 'Écouter' : 'Regarder'}
          </button>
        </div>
      )}
    </div>
  );
}

/** Copie la transcription telle qu'on la colle ailleurs — un courriel,
 *  un chatter, un document : « Interlocuteur : texte », ligne par ligne.
 *
 *  Le bouton dit lui-même que c'est fait, plutôt qu'une notice ailleurs
 *  dans la page qu'on ne regarde pas au moment de coller.
 */
function Copier({ texte, libelle = 'Copier la transcription' }) {
  const [etat, setEtat] = useState('repos');
  if (!texte?.trim()) return null;
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(texte);
          setEtat('copie');
        } catch {
          setEtat('refus');
        }
        setTimeout(() => setEtat('repos'), 2000);
      }}
      className="rounded-md px-3 py-1.5 text-[0.8125rem] text-fonce/70 ring-1 ring-bord transition-colors hover:bg-white/60 hover:text-fonce"
    >
      {etat === 'copie'
        ? 'Copiée ✓'
        : etat === 'refus'
          ? 'Copie refusée par le navigateur'
          : libelle}
    </button>
  );
}

/** Un mot ou un nom — pas une phrase : c'est ce qu'on peut corriger d'un
 *  remplacement. */
function estUnMot(texte) {
  const t = String(texte || '').trim();
  return Boolean(t) && t.split(/\s+/).length <= 3;
}

/** Toutes les occurrences d'un terme : la réplique, et le rang dans la
 *  réplique — une même phrase peut le dire deux fois. */
function occurrencesDe(segments, terme) {
  const cible = plier(terme.trim());
  if (!cible) return [];
  const liste = [];
  segments.forEach((seg, rang) => {
    const plie = plier(seg.text);
    let k = 0;
    let trouve = plie.indexOf(cible);
    while (trouve >= 0) {
      liste.push({ rang, k });
      k += 1;
      trouve = plie.indexOf(cible, trouve + cible.length);
    }
  });
  return liste;
}

/** « 17:29 » ou « 1:02:03 » → secondes ; null si illisible. */
function lireHorodatage(texte) {
  const parties = String(texte || '').trim().split(':').map(Number);
  if (!parties.length || parties.some((p) => Number.isNaN(p))) return null;
  return parties.reduce((total, p) => total * 60 + p, 0);
}

/** Ce dont le modèle n'était pas sûr.
 *
 *  L'app macOS écrivait ça dans un fichier « à vérifier » ; c'est la
 *  liste de ce qu'il faut réécouter, et elle ne vaut que si on la voit.
 */
function AVerifier({ fiche, surAller, surCorriger }) {
  // Un mot qui n'apparaît plus nulle part a été corrigé — avant que le
  // serveur ne sache retirer lui-même ce qui est réglé. Une phrase, elle,
  // peut ne pas être citée mot pour mot : on la garde.
  const texte = plier(fiche.segments.map((s) => s.text).join('\n'));
  const passages = (fiche.uncertain || []).filter(
    (p) => !estUnMot(p.text) || texte.includes(plier(p.text)),
  );
  if (!passages.length) return null;
  return (
    <div>
      <h2 className="titre text-[1.0625rem] font-medium">À vérifier</h2>
      <p className="mt-1 text-[0.8125rem] text-fonce/55">
        {passages.length} passage{passages.length > 1 ? 's' : ''} dont le modèle
        doute — un clic montre le mot dans la transcription et le propose à la
        correction.
      </p>
      <ul className="mt-2 space-y-2">
        {passages.map((p, i) => {
          const t = lireHorodatage(p.timestamp);
          const mot = estUnMot(p.text);
          return (
            <li key={i}>
              <button
                type="button"
                disabled={t === null && !mot}
                onClick={() => {
                  surAller(t, p.text);
                  if (mot) surCorriger(p.text.trim());
                }}
                className="w-full rounded-lg bg-papier p-2.5 text-left transition-colors hover:bg-turquoise/15 disabled:cursor-default disabled:hover:bg-papier"
              >
                <span className="block text-[0.75rem] tabular-nums text-turquoise-sombre">
                  {p.timestamp || '—'}
                </span>
                <span className="block text-[0.875rem]">{p.text}</span>
                {p.reason ? (
                  <span className="block text-[0.8125rem] text-fonce/55">{p.reason}</span>
                ) : null}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** Relire la transcription à la lumière du bon dossier.
 *
 *  Une liaison corrigée ne doit pas coûter une transcription : le texte
 *  existe déjà, seule sa lecture change. Titre, noms d'interlocuteurs et
 *  corrections métier sont refaits pour quelques centimes, et la version
 *  d'avant reste dans l'historique.
 */
function Relecture({ fiche, jobId, surMaj, surNote, surErreur }) {
  const [occupe, setOccupe] = useState(false);
  if (!fiche.transcript?.trim()) return null;
  return (
    <button
      type="button"
      disabled={occupe}
      title="Relit le texte déjà transcrit pour refaire le titre, les noms et les corrections métier — sans retranscrire, pour moins d'un centime."
      onClick={async () => {
        setOccupe(true);
        try {
          const vue = await api.reenrichir(jobId);
          surNote(`Relu pour ${usd(vue.cost_usd)} — ${vue.title}`);
          surMaj();
        } catch (e) { surErreur(e.message); }
        finally { setOccupe(false); }
      }}
      className="rounded-md px-3 py-1.5 text-[0.8125rem] text-fonce/70 ring-1 ring-bord transition-colors hover:bg-white/60 hover:text-fonce disabled:opacity-50"
    >
      {occupe ? 'Relecture…' : 'Refaire titre et noms'}
    </button>
  );
}

function Interlocuteurs({ fiche, jobId, surMaj, surNote, surErreur }) {
  const [carte, setCarte] = useState(fiche.speakers);
  useEffect(() => setCarte(fiche.speakers), [fiche.speakers]);
  const entrees = Object.entries(carte);
  if (entrees.length === 0) return null;

  return (
    <div>
      <h2 className="titre text-[1.0625rem] font-medium">Interlocuteurs</h2>
      <div className="mt-3 space-y-2">
        {entrees.map(([etiquette, nom]) => (
          <Champ
            key={etiquette}
            label={etiquette}
            value={nom}
            placeholder="Qui est-ce ?"
            onChange={(e) => setCarte({ ...carte, [etiquette]: e.target.value })}
          />
        ))}
      </div>
      <Bouton
        variante="discret"
        className="mt-3 w-full"
        onClick={async () => {
          try {
            await api.patch(jobId, { speakers: carte });
            surNote('Interlocuteurs enregistrés.');
            surMaj();
          } catch (e) { surErreur(e.message); }
        }}
      >
        Enregistrer
      </Bouton>
    </div>
  );
}

function Termes({ fiche, jobId, surMaj, surNote, surErreur, aCorriger }) {
  const [ancien, setAncien] = useState('');
  const [nouveau, setNouveau] = useState('');
  const champNouveau = useRef(null);

  // Un mot douteux cliqué arrive ici, prêt à corriger : il ne reste qu'à
  // taper la bonne orthographe.
  useEffect(() => {
    if (!aCorriger) return;
    setAncien(aCorriger.mot);
    setNouveau('');
    champNouveau.current?.focus({ preventScroll: true });
  }, [aCorriger]);

  return (
    <div>
      <h2 className="titre text-[1.0625rem] font-medium">Vocabulaire</h2>
      {fiche.technical_terms.length ? (
        <p className="mt-2 text-[0.875rem] leading-relaxed text-fonce/70">
          {fiche.technical_terms.join(' · ')}
        </p>
      ) : (
        <p className="mt-2 text-[0.875rem] text-fonce/50">Aucun terme relevé.</p>
      )}

      <h3 className="titre mt-5 text-[0.9375rem] font-medium">Corriger un mot mal entendu</h3>
      {/* Une phrase à compléter plutôt que deux champs empilés : le sens
          du remplacement se lit, il ne se devine pas. */}
      <div className="mt-2 grid grid-cols-[1fr_auto_1fr] items-end gap-2">
        <Champ
          label="Mot transcrit (faux)"
          placeholder="Acritek"
          value={ancien}
          onChange={(e) => setAncien(e.target.value)}
        />
        <span aria-hidden className="pb-2 text-fonce/40">→</span>
        <Champ
          ref={champNouveau}
          label="Bonne orthographe"
          placeholder="Acritec"
          value={nouveau}
          onChange={(e) => setNouveau(e.target.value)}
        />
      </div>
      <p className="mt-2 text-[0.8125rem] text-fonce/55">
        {ancien.trim() && nouveau.trim()
          ? <>« {ancien.trim()} » deviendra « {nouveau.trim()} » partout : transcription, segments et vocabulaire.</>
          : 'Le remplacement s’applique à toute la réunion, et le bon mot rejoint le vocabulaire de l’équipe.'}
      </p>
      <Bouton
        variante="discret"
        className="mt-3 w-full"
        disabled={!ancien.trim() || !nouveau.trim()}
        onClick={async () => {
          try {
            const vue = await api.replaceTerm(jobId, ancien, nouveau);
            surNote(
              vue.occurrences
                ? `${vue.occurrences} occurrence${vue.occurrences > 1 ? 's' : ''} corrigée${vue.occurrences > 1 ? 's' : ''}.`
                : 'Aucune occurrence trouvée.',
            );
            setAncien(''); setNouveau('');
            surMaj();
          } catch (e) { surErreur(e.message); }
        }}
      >
        Remplacer partout
      </Bouton>
    </div>
  );
}

/** Relance ciblée.
 *
 *  Sur une réunion de huit fenêtres dont trois ont échoué, relancer les
 *  cinq bonnes serait les repayer pour rien.
 */
function Fenetres({ jobId, surNote, surErreur }) {
  const [vue, setVue] = useState(null);
  useEffect(() => { api.job(jobId).then(setVue).catch(() => {}); }, [jobId]);
  if (!vue || vue.chunks.length <= 1) return null;

  return (
    <div>
      <h2 className="titre text-[1.0625rem] font-medium">Fenêtres</h2>
      <ul className="verre mt-3 divide-y divide-bord/60 rounded-xl">
        {vue.chunks.map((c) => (
          <li key={c.index} className="flex items-center gap-3 px-3 py-2 text-[0.875rem]">
            <span className="w-12 shrink-0 tabular-nums text-fonce/45">{horodatage(c.start)}</span>
            <span className={`flex-1 ${c.status === 'erreur' ? 'text-[#8c1d18]' : 'text-fonce/70'}`}>
              {c.status === 'termine' ? 'terminée' : c.status === 'erreur' ? 'erreur' : c.status}
            </span>
            <button
              className="text-turquoise-sombre hover:underline"
              onClick={async () => {
                try {
                  await api.resetChunk(jobId, c.index);
                  surNote(`Fenêtre ${c.index + 1} à refaire — relance-la depuis l'onglet de transcription.`);
                  setVue(await api.job(jobId));
                } catch (e) { surErreur(e.message); }
              }}
            >
              relancer
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Les versions précédentes, lisibles.
 *
 *  Une relance ou une relecture empile la version d'avant plutôt que de
 *  l'écraser. Encore faut-il pouvoir la lire : chacune se déplie sur son
 *  texte complet, qu'on peut copier pour reprendre un passage.
 */
/** Les versions antérieures, sous la transcription.
 *
 *  Discrètes : une ligne repliée, qu'on ouvre au besoin. Chaque relance
 *  ou relecture en empile une ; on peut la lire, la copier, ou y revenir
 *  — et revenir en arrière se défait de la même façon, la version en
 *  place rejoignant l'historique.
 */
function Versions({ versions, jobId, surMaj, surNote, surErreur }) {
  const [ouvert, setOuvert] = useState(false);
  const [lue, setLue] = useState(null);
  const [aConfirmer, setAConfirmer] = useState(null);
  const [occupe, setOccupe] = useState(false);
  if (!versions?.length) return null;

  const revenir = async (rang) => {
    setOccupe(true);
    try {
      await api.restaurerVersion(jobId, rang);
      surNote(`Version du ${jour(versions[rang].archived_at)} remise en place — celle d'avant est dans l'historique.`);
      setAConfirmer(null);
      setLue(null);
      surMaj();
    } catch (e) { surErreur(e.message); }
    finally { setOccupe(false); }
  };

  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOuvert((o) => !o)}
        aria-expanded={ouvert}
        className="text-[0.8125rem] text-fonce/50 hover:text-fonce"
      >
        {ouvert ? '▾' : '▸'} {versions.length} version{versions.length > 1 ? 's' : ''} antérieure{versions.length > 1 ? 's' : ''}
      </button>
      {ouvert ? (
        <ul className="mt-2 space-y-1.5">
          {versions.map((v, rang) => (
            <li key={v.archived_at + rang} className="rounded-lg bg-white/50 px-3 py-2">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="text-[0.75rem] tabular-nums text-fonce/50">{jour(v.archived_at)}</span>
                <span className="min-w-0 flex-1 truncate text-[0.875rem]">{v.title || 'Sans titre'}</span>
                <span className="flex shrink-0 gap-3 text-[0.8125rem]">
                  <button type="button" onClick={() => setLue(lue === rang ? null : rang)}
                          className="text-violet hover:underline">
                    {lue === rang ? 'replier' : 'lire'}
                  </button>
                  <button
                    type="button"
                    disabled={!v.restaurable}
                    onClick={() => setAConfirmer(rang)}
                    title={v.restaurable
                      ? 'Remettre cette version en place'
                      : 'Seul le texte de cette version a été gardé : copie-le si besoin.'}
                    className="text-fonce/60 hover:text-fonce hover:underline disabled:cursor-not-allowed disabled:text-fonce/30 disabled:no-underline"
                  >
                    y revenir
                  </button>
                </span>
              </div>
              {aConfirmer === rang ? (
                <div className="mt-2 flex flex-wrap items-center gap-2 rounded-md bg-papier px-3 py-2 text-[0.8125rem]">
                  <span className="flex-1 text-fonce/70">
                    Remettre cette version en place ? La version actuelle sera gardée dans l'historique.
                  </span>
                  <button type="button" disabled={occupe} onClick={() => revenir(rang)}
                          className="rounded-md bg-fonce px-3 py-1 text-clair hover:bg-fonce-doux disabled:opacity-50">
                    {occupe ? 'Restauration…' : 'Y revenir'}
                  </button>
                  <button type="button" onClick={() => setAConfirmer(null)}
                          className="px-2 py-1 text-fonce/55 hover:text-fonce">
                    Annuler
                  </button>
                </div>
              ) : null}
              {lue === rang ? (
                <>
                  <div className="mt-2 max-h-72 overflow-y-auto whitespace-pre-wrap rounded-md bg-white/80 p-2 text-[0.8125rem] leading-relaxed text-fonce/80">
                    {v.transcript || 'Texte non conservé pour cette version.'}
                  </div>
                  <div className="mt-2 flex justify-end">
                    <Copier texte={v.transcript} libelle="Copier cette version" />
                  </div>
                </>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

/** Dépôt dans le chatter Odoo.
 *
 *  Remplace la recopie manuelle qui a fini par gonfler des opportunités
 *  jusqu'à 300 000 caractères. La transcription part en accordéon : le
 *  texte intégral reste disponible sans noyer l'historique commercial.
 */
function Odoo({ fiche, jobId, surMaj, surNote, surErreur }) {
  const [requete, setRequete] = useState('');
  const [candidats, setCandidats] = useState(null);
  const [envoi, setEnvoi] = useState(false);
  // Relire puis déposer prend une vingtaine de secondes : sans étape
  // affichée, l'écran paraissait figé alors que tout se passait bien.
  const [etape, setEtape] = useState('');
  const lien = fiche.odoo || {};

  useEffect(() => {
    const terme = requete.trim();
    if (terme.length < 2) { setCandidats(null); return undefined; }
    const attente = setTimeout(() => {
      api.odooRecords(terme).then((v) => setCandidats(v.records)).catch(() => setCandidats([]));
    }, 300);
    return () => clearTimeout(attente);
  }, [requete]);

  if (lien.message_id) {
    return (
      <div>
        <h2 className="titre text-[1.0625rem] font-medium">Odoo</h2>
        <p className="mt-2 text-[0.875rem] text-turquoise-sombre">
          Déposée dans le chatter le {jour(lien.published_at)}.
        </p>
        <p className="mt-0.5 text-[0.8125rem] text-fonce/50">
          {lien.model} #{lien.record_id}
        </p>
      </div>
    );
  }

  if (!fiche.transcript?.trim()) return null;

  return (
    <div>
      <h2 className="titre text-[1.0625rem] font-medium">Odoo</h2>
      {lien.record_id ? (
        <div className="mt-2 rounded-lg bg-papier p-3">
          <p className="text-[0.875rem]">
            Dossier retenu avant la transcription :{' '}
            <span className="titre font-medium">{lien.model} #{lien.record_id}</span>
          </p>
          <Bouton
            className="mt-2"
            disabled={envoi}
            onClick={async () => {
              setEnvoi(true);
              try {
                await api.odooPublish(jobId, {
                  model: lien.model, record_id: lien.record_id,
                });
                surNote('Déposée dans le dossier retenu.');
                surMaj();
              } catch (e) { surErreur(e.message); }
              finally { setEnvoi(false); }
            }}
          >
            Déposer ici
          </Bouton>
        </div>
      ) : null}
      <p className="mt-3 text-[0.8125rem] text-fonce/55">
        {lien.record_id
          ? 'Ou chercher un autre dossier.'
          : 'Déposer la transcription dans le chatter, repliée en accordéon.'}
      </p>
      <div className="mt-2">
        <Champ
          label="Chercher l'opportunité"
          placeholder="Acritec, Canolle…"
          value={requete}
          onChange={(e) => setRequete(e.target.value)}
        />
      </div>

      {etape ? (
        <p className="mt-2 flex items-center gap-2 text-[0.8125rem] text-turquoise-sombre">
          <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-turquoise border-t-transparent" />
          {etape}
        </p>
      ) : null}

      {candidats?.length === 0 ? (
        <p className="mt-2 text-[0.8125rem] text-fonce/50">Aucun dossier trouvé.</p>
      ) : null}

      {candidats?.length ? (
        <ul className="mt-2 space-y-1">
          {candidats.map((c) => (
            <li key={`${c.model}-${c.id}`}>
              <button
                type="button"
                disabled={envoi}
                onClick={async () => {
                  setEnvoi(true);
                  try {
                    // Relire d'abord : le dossier qu'on vient de choisir
                    // change le titre et les noms, et la note déposée
                    // doit porter la bonne version.
                    setEtape(`Lecture du dossier « ${c.name} » et relecture du texte…`);
                    const relu = await api
                      .reenrichir(jobId, { model: c.model, record_id: c.id })
                      .catch(() => null);
                    setEtape('Dépôt dans le chatter…');
                    await api.odooPublish(jobId, { model: c.model, record_id: c.id });
                    surNote(relu
                      ? `Relue puis déposée dans « ${c.name} ».`
                      : `Déposée dans « ${c.name} ».`);
                    surMaj();
                  } catch (e) { surErreur(e.message); }
                  finally { setEnvoi(false); setEtape(''); }
                }}
                className="w-full rounded-md px-2 py-1.5 text-left text-[0.875rem] transition-colors hover:bg-white/60 disabled:opacity-40"
              >
                <span className="titre font-medium">{c.name}</span>
                <span className="block text-[0.8125rem] text-fonce/50">
                  {c.partner || c.model} · {c.updated}
                </span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
