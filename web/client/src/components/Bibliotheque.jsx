import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api.js';
import { Bouton, Champ, Etat, Erreur, Vide } from './Communs.jsx';
import { duree, fileSize, usd, jour, horodatage } from '../format.js';
import { MOD, useRaccourcis } from '../raccourcis.js';
import { surlignerMots } from '../surlignage.jsx';
import { Welcome } from './Welcome.jsx';
import { interrompreReunion } from '../usePipeline.js';
import {
  enqueue, isMedia, launch, remoteFile, removeFromQueue, requestLaunch, useFileDrag, useFileQueue,
} from '../fileQueue.js';

/** La bibliothèque : une table, pas une grille de cartes.
 *
 *  Ce sont des lignes comparables qu'on trie et qu'on balaie ; les
 *  encadrer une à une ajouterait des contenants sans rien clarifier.
 */
const PAR_PAGE = 50;
const COLONNES_SERVEUR = {
  quand: 'date', title: 'title', duration_seconds: 'duration', status: 'status', cost_usd: 'cost',
};

export function Bibliotheque({ surOuvrir, surLancer, surVue }) {
  const [jobs, setJobs] = useState(null);
  const [erreur, setErreur] = useState('');
  const [tri, setTri] = useState({ champ: 'quand', sens: 'desc' });
  const [recherche, setRecherche] = useState('');
  const [resultats, setResultats] = useState(null);
  const [etat, setEtat] = useState('actif');
  const [retention, setRetention] = useState(30);
  const champRecherche = useRef(null);

  // Sélection multiple : ⌥-clic (ou ⌘/Ctrl-clic) ajoute ou retire une
  // ligne, Maj-clic prend toute la plage depuis la dernière touchée. Tant
  // qu'il y a une sélection, un simple clic la complète au lieu d'ouvrir.
  const [choisis, setChoisis] = useState(() => new Set());
  const ancre = useRef(null);

  const chercher = () => { champRecherche.current?.focus(); champRecherche.current?.select(); };
  const vider = () => { setChoisis(new Set()); ancre.current = null; };

  // Le tri et la pagination se font au serveur : la bibliothèque peut
  // compter des centaines de réunions, et le navigateur n'en reçoit
  // qu'une page.
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const recharger = useCallback(() => {
    api.listJobs(etat, { sort: COLONNES_SERVEUR[tri.champ] || 'date', order: tri.sens, page, perPage: PAR_PAGE })
      .then(({ items, total: n }) => { setJobs(items); setTotal(n); })
      .catch((e) => setErreur(e.message));
  }, [etat, tri, page]);

  useEffect(() => { setPage(1); }, [etat, tri]);
  useEffect(() => { setJobs(null); recharger(); }, [recharger]);

  // Tant qu'une réunion tourne, la liste se rafraîchit seule : on doit
  // pouvoir lancer une deuxième transcription et regarder la première
  // avancer d'ici.
  const tourne = (jobs || []).some((j) => j.progress && j.status !== 'erreur');
  useEffect(() => {
    if (!tourne) return undefined;
    const minuteur = setInterval(recharger, 4000);
    return () => clearInterval(minuteur);
  }, [tourne, recharger]);

  useEffect(() => {
    api.settings()
      .then((r) => setRetention(r.corbeille?.retention_jours ?? 30))
      .catch(() => {});
  }, []);

  // La recherche plein texte vit côté serveur (FTS5) : on ne filtre pas
  // la table en local, sinon elle ne verrait que la page chargée.
  useEffect(() => {
    const terme = recherche.trim();
    if (terme.length < 2) { setResultats(null); return undefined; }
    const attente = setTimeout(() => {
      api.search(terme).then(setResultats).catch((e) => setErreur(e.message));
    }, 250);
    return () => clearTimeout(attente);
  }, [recherche]);

  useEffect(() => { vider(); }, [etat]);
  // Une réunion sortie de la liste (jetée, archivée) sort de la sélection.
  useEffect(() => {
    if (!jobs) return;
    setChoisis((c) => {
      const presents = new Set(jobs.map((j) => j.job_id));
      const reste = new Set([...c].filter((id) => presents.has(id)));
      return reste.size === c.size ? c : reste;
    });
  }, [jobs]);

  // Déjà dans l'ordre du serveur ; la date affichée est celle de la
  // réunion, à défaut celle du dépôt.
  const triees = useMemo(
    () => (jobs || []).map((j) => ({ ...j, quand: j.meeting_date || j.created_at })),
    [jobs],
  );

  useRaccourcis({
    'mod+f': chercher,
    '/': chercher,
    'mod+a': (e) => {
      // Dans un champ, ⌘A sélectionne le texte, comme partout.
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return false;
      if (resultats || !triees.length) return false;
      setChoisis(new Set(triees.map((j) => j.job_id)));
      return undefined;
    },
    escape: () => (choisis.size ? vider() : false),
  });

  const cliquerLigne = (evenement, id) => {
    const ordre = triees.map((j) => j.job_id);
    if (evenement.shiftKey) {
      const depart = ordre.indexOf(ancre.current);
      const arrivee = ordre.indexOf(id);
      const [de, a] = depart < 0 ? [arrivee, arrivee] : [Math.min(depart, arrivee), Math.max(depart, arrivee)];
      setChoisis((c) => new Set([...c, ...ordre.slice(de, a + 1)]));
      if (depart < 0) ancre.current = id;
      return;
    }
    if (evenement.altKey || evenement.metaKey || evenement.ctrlKey || choisis.size) {
      setChoisis((c) => {
        const suite = new Set(c);
        if (suite.has(id)) suite.delete(id); else suite.add(id);
        return suite;
      });
      ancre.current = id;
      return;
    }
    surOuvrir(id);
  };

  const colonne = (champ, libelle, classe = '') => (
    <th className={`px-4 py-2 text-left text-ekn-sm font-medium text-fonce/60 ${classe}`}>
      <button
        onClick={() => setTri((t) => {
          // Un titre se lit de A à Z ; le reste, du plus récent ou du plus gros.
          const premier = champ === 'title' ? 'asc' : 'desc';
          const autre = premier === 'asc' ? 'desc' : 'asc';
          return { champ, sens: t.champ === champ && t.sens === premier ? autre : premier };
        })}
        className="hover:text-fonce"
      >
        {libelle}
        {tri.champ === champ ? <span aria-hidden> {tri.sens === 'desc' ? '↓' : '↑'}</span> : null}
      </button>
    </th>
  );

  return (
    <section className="mx-auto max-w-6xl px-6 py-10">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <h1 className="titre text-[1.5rem] font-semibold">Bibliothèque</h1>
        <div className="w-full sm:w-80">
          <Champ
            ref={champRecherche}
            label="Rechercher dans les transcriptions"
            placeholder="un mot, un nom, une décision…"
            value={recherche}
            onChange={(e) => setRecherche(e.target.value)}
          />
        </div>
      </div>

      <div className="mt-4 flex gap-1 text-ekn-sm">
        {[
          ['actif', 'Bibliothèque'],
          ['archive', 'Archives'],
          ['corbeille', 'Corbeille'],
        ].map(([cle, libelle]) => (
          <button
            key={cle}
            type="button"
            onClick={() => setEtat(cle)}
            className={`rounded-md px-3 py-1.5 transition-colors ${
              etat === cle ? 'bg-white/60 font-medium' : 'text-ekn-text-muted hover:bg-white/35'
            }`}
          >
            {libelle}
          </button>
        ))}
      </div>
      {etat === 'corbeille' ? (
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
          <p className="text-ekn-sm text-ekn-text-muted">
            Les réunions jetées disparaissent définitivement au bout de{' '}
            {retention} jours.
          </p>
          {jobs?.length ? (
            <button
              type="button"
              onClick={async () => {
                // Ici, et seulement ici, on confirme : c'est le seul
                // geste de la bibliothèque qui ne se défait pas.
                if (!window.confirm(
                  `Supprimer définitivement ${total} réunion(s) ? `
                  + 'Cette fois, rien ne sera récupérable.',
                )) return;
                try { await api.viderCorbeille(); recharger(); }
                catch (e) { setErreur(e.message); }
              }}
              className="rounded-md px-3 py-1.5 text-ekn-sm text-violet ring-1 ring-violet/35 hover:bg-white/50"
            >
              Vider la corbeille
            </button>
          ) : null}
        </div>
      ) : null}

      <Erreur>{erreur}</Erreur>

      {etat === 'actif' ? <Welcome onNavigate={(vue) => surVue?.(vue)} /> : null}
      {etat === 'actif' ? <ALancer surLancer={surLancer} /> : null}
      <DeposerIci />

      {resultats ? (
        <Resultats resultats={resultats} requete={recherche} surOuvrir={(id) => surOuvrir(id, recherche.trim())} />
      ) : jobs === null ? (
        <p className="py-16 text-center text-ekn-text-muted">Chargement…</p>
      ) : jobs.length === 0 ? (
        etat === 'corbeille' ? (
          <Vide titre="Corbeille vide">Rien à récupérer.</Vide>
        ) : etat === 'archive' ? (
          <Vide titre="Aucune archive">
            Archiver sort une réunion de la bibliothèque sans la perdre : elle
            reste trouvable par la recherche.
          </Vide>
        ) : (
          <Vide titre="Aucune transcription pour l'instant">
            Lance-en une depuis l'onglet « Nouvelle transcription ». Ton fichier
            restera sur ton poste.
          </Vide>
        )
      ) : (
        <div className="verre mt-6 overflow-x-auto rounded-xl">
          <table className="w-full min-w-[46rem] border-collapse">
            <thead className="border-b border-bord">
              <tr>
                {colonne('title', 'Réunion')}
                {colonne('quand', 'Date', 'whitespace-nowrap')}
                {colonne('duration_seconds', 'Durée')}
                {colonne('status', 'État')}
                {colonne('cost_usd', 'Coût', 'text-right')}
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody>
              {triees.map((job) => (
                <tr
                  key={job.job_id}
                  onClick={(e) => cliquerLigne(e, job.job_id)}
                  // Maj-clic et ⌥-clic sélectionneraient aussi du texte.
                  onMouseDown={(e) => { if (e.shiftKey || e.altKey) e.preventDefault(); }}
                  aria-selected={choisis.has(job.job_id)}
                  className={`cursor-pointer border-b border-bord/50 last:border-0 transition-colors ${
                    choisis.has(job.job_id)
                      ? 'bg-turquoise/25 shadow-[inset_3px_0_0_var(--color-turquoise-sombre)] hover:bg-turquoise/30'
                      : 'hover:bg-white/45'
                  }`}
                >
                  <td className="px-4 py-3">
                    <span className="titre font-medium">{job.title || job.filename}</span>
                    {/* Sur la ligne du fichier plutôt qu'à la suite du titre :
                        derrière un titre long, l'étiquette passait seule à la
                        ligne et flottait. */}
                    {job.title || job.has_versions ? (
                      <span className="block text-ekn-sm text-ekn-text-muted">
                        {job.title ? job.filename : null}
                        {job.title && job.has_versions ? ' · ' : null}
                        {job.has_versions ? (
                          <span className="whitespace-nowrap text-violet">version antérieure</span>
                        ) : null}
                      </span>
                    ) : null}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-fonce/70">{jour(job.quand)}</td>
                  <td className="px-4 py-3 tabular-nums text-fonce/70">{duree(job.duration_seconds)}</td>
                  <td className="px-4 py-3">
                    <Etat valeur={job.status} />
                    {job.progress && job.status !== 'erreur' && job.progress.total ? (
                      <span className="mt-1 block w-24">
                        <span className="block h-1 overflow-hidden rounded-full bg-bord">
                          <span
                            className="block h-full bg-turquoise transition-[width]"
                            style={{ width: `${(job.progress.done / job.progress.total) * 100}%` }}
                          />
                        </span>
                        <span className="text-ekn-xs tabular-nums text-ekn-text-muted">
                          {job.progress.done}/{job.progress.total} fenêtres
                        </span>
                      </span>
                    ) : null}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums text-fonce/70">{usd(job.cost_usd)}</td>
                  <td
                    className="whitespace-nowrap px-4 py-3 text-right"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Actions
                      job={job}
                      etat={etat}
                      surFait={recharger}
                      surErreur={setErreur}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {jobs !== null && total > PAR_PAGE && !resultats ? (
        <Pagination
          page={page}
          total={total}
          parPage={PAR_PAGE}
          surPage={(numero) => { setPage(numero); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
        />
      ) : null}
      {choisis.size && !resultats ? (
        <Selection
          jobs={triees.filter((j) => choisis.has(j.job_id))}
          etat={etat}
          total={triees.length}
          surTout={() => setChoisis(new Set(triees.map((j) => j.job_id)))}
          surVider={vider}
          surFait={recharger}
          surErreur={setErreur}
          surLancer={surLancer}
        />
      ) : null}
    </section>
  );
}

/** Ce qu'on peut faire d'une sélection.
 *
 *  Une barre flottante, qui n'apparaît qu'avec la sélection : les mêmes
 *  gestes que sur une ligne, appliqués à toutes. Les réunions partent
 *  une à une, et l'avancement se voit.
 */
function Selection({ jobs, etat, total, surTout, surVider, surFait, surErreur, surLancer }) {
  const [cours, setCours] = useState(null); // { libelle, fait, total }
  const [bilan, setBilan] = useState('');

  const appliquer = async (libelle, action, cibles = jobs) => {
    setBilan('');
    const echecs = [];
    setCours({ libelle, fait: 0, total: cibles.length });
    for (const [rang, job] of cibles.entries()) {
      try { await action(job.job_id); } catch (e) { echecs.push(e.message); }
      setCours({ libelle, fait: rang + 1, total: cibles.length });
    }
    setCours(null);
    if (echecs.length) surErreur(`${echecs.length} réunion(s) en échec : ${echecs[0]}`);
    surFait();
    return cibles.length - echecs.length;
  };

  const n = jobs.length;
  const pluriel = n > 1 ? 's' : '';
  const terminees = jobs.filter((j) => j.status === 'termine');
  const aRetraiter = jobs.filter((j) => j.reprocessable && j.video_bytes);

  // Retraiter en lot : tout passe par la file de « Nouvelle
  // transcription », une réunion après l'autre, comme des fichiers déposés.
  const retraiter = () => {
    const [premiere, ...suite] = aRetraiter.map(remoteFile);
    enqueue(suite);
    launch(premiere);
    surVider();
    surLancer?.();
  };

  const bouton = (libelle, surClic, titre, danger = false) => (
    <button
      type="button"
      title={titre}
      disabled={Boolean(cours)}
      onClick={surClic}
      className={`rounded-md px-2.5 py-1 text-ekn-sm transition-colors hover:bg-white/10 disabled:opacity-40 ${
        danger ? 'text-ekn-error-light' : 'text-clair'
      }`}
    >
      {libelle}
    </button>
  );

  return (
    <div
      role="toolbar"
      aria-label="Actions sur la sélection"
      className="fixed inset-x-0 bottom-6 z-40 mx-auto flex w-fit max-w-[calc(100vw-2rem)] flex-wrap items-center gap-1 rounded-xl bg-fonce ekn-dark px-3 py-2 text-clair shadow-2xl"
    >
      <span className="px-2 text-ekn-sm tabular-nums text-clair/70">
        {cours
          ? `${cours.libelle} ${cours.fait}/${cours.total}…`
          : bilan || `${n} sélectionnée${pluriel}`}
      </span>
      <span className="mx-1 h-4 w-px bg-clair/20" />
      {etat === 'actif' ? (
        <>
          {bouton('Archiver', () => appliquer('Archivage', api.archiver), 'Sortir de la bibliothèque, sans perdre')}
          {bouton('Jeter', () => appliquer('Mise à la corbeille', api.jeter), 'Mettre à la corbeille — récupérable')}
          {bouton(
            'Refaire titre et noms',
            async () => {
              const faites = await appliquer('Relecture', (id) => api.reenrichir(id), terminees);
              setBilan(`${faites} relue${faites > 1 ? 's' : ''}`);
            },
            terminees.length < n
              ? `Relit le texte déjà transcrit — seules les ${terminees.length} réunion(s) terminées sont concernées.`
              : 'Relit le texte déjà transcrit pour refaire titre, noms et corrections — moins d’un centime chacune.',
          )}
          {aRetraiter.length
            ? bouton(
              aRetraiter.length < n ? `Retraiter (${aRetraiter.length})` : 'Retraiter',
              retraiter,
              'Transcrire les réunions qui n’ont que leur vidéo, depuis la vidéo stockée — sans dépôt Odoo d’office.',
            )
            : null}
        </>
      ) : null}
      {etat === 'archive' ? (
        <>
          {bouton('Restaurer', () => appliquer('Restauration', api.restaurer), 'Remettre dans la bibliothèque')}
          {bouton('Jeter', () => appliquer('Mise à la corbeille', api.jeter), 'Mettre à la corbeille — récupérable')}
        </>
      ) : null}
      {etat === 'corbeille' ? (
        <>
          {bouton('Restaurer', () => appliquer('Restauration', api.restaurer), 'Remettre dans la bibliothèque')}
          {bouton(
            'Supprimer définitivement',
            () => {
              if (!window.confirm(`Supprimer définitivement ${n} réunion${pluriel} ? Rien ne sera récupérable.`)) return;
              appliquer('Suppression', api.supprimerDefinitivement);
            },
            'Irréversible',
            true,
          )}
        </>
      ) : null}
      <span className="mx-1 h-4 w-px bg-clair/20" />
      {n < total ? bouton(`Tout (${total})`, surTout, `Tout sélectionner (${MOD} A)`) : null}
      <button
        type="button"
        onClick={surVider}
        aria-label="Annuler la sélection"
        title="Annuler la sélection (Échap)"
        className="rounded-md px-2 text-[1.125rem] leading-none text-clair/60 hover:bg-white/10 hover:text-clair"
      >
        ×
      </button>
    </div>
  );
}

/** Jeter, archiver, restaurer — sans quitter la liste.
 *
 *  Pas de confirmation avant de jeter : la corbeille *est* la
 *  confirmation, et elle se défait d'un clic. Demander deux fois pour un
 *  geste réversible ne protège de rien et use l'attention. La
 *  suppression définitive, elle, se confirme : elle ne se défait pas.
 */
function Actions({ job, etat, surFait, surErreur }) {
  const [occupe, setOccupe] = useState(false);

  const agir = async (action) => {
    setOccupe(true);
    try { await action(job.job_id); surFait(); }
    catch (e) { surErreur(e.message); }
    finally { setOccupe(false); }
  };

  const bouton = (libelle, action, titre) => (
    <button
      type="button"
      title={titre}
      disabled={occupe}
      onClick={() => agir(action)}
      className="rounded px-2 py-1 text-ekn-sm text-ekn-text-muted transition-colors hover:bg-white/60 hover:text-fonce disabled:opacity-40"
    >
      {libelle}
    </button>
  );

  if (etat === 'actif' && ['en_attente', 'en_cours', 'a_finaliser'].includes(job.status)) {
    // En cours, ou bloquée : on l'arrête plutôt que de l'archiver.
    return bouton('Interrompre', async (id) => {
      if (!window.confirm('Interrompre cette transcription ? La réunion part à la corbeille.')) {
        throw new Error('Interruption annulée.');
      }
      return interrompreReunion(id);
    }, 'Arrêter la transcription — la réunion part à la corbeille');
  }
  if (etat === 'actif') {
    return (
      <>
        {bouton('Archiver', api.archiver, 'Sortir de la bibliothèque, sans perdre')}
        {bouton('Jeter', api.jeter, 'Mettre à la corbeille')}
      </>
    );
  }
  if (etat === 'corbeille') {
    return (
      <>
        {bouton('Restaurer', api.restaurer, 'Remettre dans la bibliothèque')}
        {bouton('Supprimer', async (id) => {
          if (!window.confirm('Supprimer définitivement cette réunion ?')) {
            throw new Error('Suppression annulée.');
          }
          return api.supprimerDefinitivement(id);
        }, 'Supprimer définitivement — irréversible')}
      </>
    );
  }
  return bouton('Restaurer', api.restaurer, 'Remettre dans la bibliothèque');
}

function Resultats({ resultats, requete, surOuvrir }) {
  if (resultats.length === 0) {
    return <Vide titre="Rien trouvé">Aucun passage ne contient ces mots.</Vide>;
  }
  return (
    <ul className="verre mt-6 divide-y divide-bord/60 rounded-xl">
      {resultats.map((hit, rang) => (
        <li key={`${hit.job_id}-${rang}`}>
          <button
            onClick={() => surOuvrir(hit.job_id)}
            className="block w-full px-4 py-3 text-left transition-colors hover:bg-white/45"
          >
            <span className="text-ekn-sm text-ekn-text-muted">
              {hit.title || hit.filename} · {horodatage(hit.start_second)}
              {hit.speaker ? ` · ${hit.speaker}` : ''}
            </span>
            <span className="mt-0.5 block">{surlignerMots(hit.text, requete)}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}

/** Les fichiers déposés qui attendent d'être lancés.
 *
 *  Ils restent sur le poste : la liste vit dans l'onglet, et le dit.
 *  Chacun se lance quand on veut, dans l'ordre qu'on veut.
 */
function ALancer({ surLancer }) {
  const enAttente = useFileQueue();
  if (!enAttente.length) return null;
  return (
    <div className="verre mt-6 rounded-xl p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="titre text-[1.0625rem] font-medium">
          À lancer <span className="tabular-nums text-ekn-text-muted">{enAttente.length}</span>
        </h2>
        <span className="text-ekn-sm text-ekn-text-muted">
          Restés sur ton poste : la liste se vide si tu fermes l’onglet.
        </span>
      </div>
      <ul className="mt-2 divide-y divide-bord/60">
        {enAttente.map((entree) => (
          <li key={entree.id} className="flex items-center gap-3 py-2 text-ekn-sm">
            <span className="min-w-0 flex-1 truncate">{entree.file.name}</span>
            <span className="whitespace-nowrap text-ekn-text-muted">{jour(new Date(entree.file.lastModified).toISOString())}</span>
            <span className="w-16 whitespace-nowrap text-right tabular-nums text-ekn-text-muted">{fileSize(entree.file.size)}</span>
            <button
              type="button"
              onClick={() => { requestLaunch(entree.id); surLancer?.(); }}
              className="rounded-md bg-fonce px-3 py-1 text-ekn-sm text-clair hover:bg-fonce-doux"
            >
              Lancer
            </button>
            <button
              type="button"
              onClick={() => removeFromQueue(entree.id)}
              className="px-1 text-ekn-sm text-ekn-text-muted hover:text-fonce"
            >
              Retirer
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Déposer des fichiers sur la bibliothèque les ajoute à « À lancer ». */
function DeposerIci() {
  const glisse = useFileDrag();
  const [message, setMessage] = useState('');
  useEffect(() => {
    if (!message) return undefined;
    const minuteur = setTimeout(() => setMessage(''), 4000);
    return () => clearTimeout(minuteur);
  }, [message]);

  return (
    <>
      {glisse ? (
        <div
          className="fixed inset-0 z-40 flex items-center justify-center bg-fonce/25 p-6 backdrop-blur-[2px]"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const liste = [...e.dataTransfer.files];
            const medias = liste.filter(isMedia);
            const ajoutes = enqueue(medias);
            const ignores = liste.length - medias.length;
            setMessage(
              `${ajoutes} fichier${ajoutes > 1 ? 's' : ''} ajouté${ajoutes > 1 ? 's' : ''} à « À lancer »`
              + (ignores ? ` · ${ignores} ignoré${ignores > 1 ? 's' : ''} (ni audio ni vidéo)` : ''),
            );
          }}
        >
          <div className="pointer-events-none rounded-2xl border-2 border-dashed border-turquoise bg-white/90 px-10 py-8 text-center shadow-2xl">
            <p className="titre text-[1.0625rem] font-medium">Dépose tes enregistrements</p>
            <p className="mt-1 text-ekn-sm text-fonce/60">Ils rejoignent « À lancer », à transcrire quand tu veux.</p>
          </div>
        </div>
      ) : null}
      {message ? (
        <p className="fixed bottom-6 left-0 right-0 z-40 mx-auto w-fit rounded-lg bg-fonce px-4 py-2 text-ekn-sm text-clair shadow-xl">
          {message}
        </p>
      ) : null}
    </>
  );
}

/** Les pages de la bibliothèque : où l'on est, combien il y en a, et de
 *  quoi passer à la voisine. */
function Pagination({ page, total, parPage, surPage }) {
  const pages = Math.ceil(total / parPage);
  const debut = (page - 1) * parPage + 1;
  const fin = Math.min(page * parPage, total);
  const bouton = (libelle, cible, actif) => (
    <button
      type="button"
      disabled={!actif}
      onClick={() => surPage(cible)}
      className="rounded-md px-3 py-1.5 text-ekn-sm text-fonce/70 ring-1 ring-bord transition-colors hover:bg-white/60 disabled:opacity-30"
    >
      {libelle}
    </button>
  );
  return (
    <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-ekn-sm">
      <span className="tabular-nums text-ekn-text-muted">{debut}–{fin} sur {total} réunions</span>
      <div className="flex items-center gap-2">
        {bouton('← Précédente', page - 1, page > 1)}
        <span className="px-2 tabular-nums text-fonce/60">Page {page} / {pages}</span>
        {bouton('Suivante →', page + 1, page < pages)}
      </div>
    </div>
  );
}
