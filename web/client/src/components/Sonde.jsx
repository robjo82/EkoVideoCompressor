import { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';

const CERTITUDE = {
  certaine: 'Liaison certaine',
  probable: 'Liaison probable',
  incertaine: 'Liaison incertaine',
};

/** Ce que la sonde a entendu, et les dossiers que ça désigne.
 *
 *  Un seul point d'arrêt dans le flux, et il est ici : lier une
 *  transcription au mauvais dossier la dépose chez un autre client,
 *  visible par toute l'équipe. Le reste peut tourner seul ; ce choix-là
 *  se valide — et se défait — d'un clic.
 */
export function Sonde({ etat, retenu, surChoix, surRetrait, surAjout }) {
  const [discussion, setDiscussion] = useState(false);
  // La conversation survit à la fermeture de la fenêtre : la rouvrir,
  // c'est reprendre où on en était, pas tout réexpliquer.
  const [echanges, setEchanges] = useState([]);

  if (!etat) return null;
  if (etat.enCours) return <Attente phase={etat.phase} />;

  const indices = etat.clues || {};
  const enquete = etat.investigation || {};
  const candidats = etat.candidates || [];
  const entendu = [...(indices.organisations || []), ...(indices.personnes || [])];
  if (!entendu.length && !candidats.length && !etat.odoo) return null;

  const estRetenu = (d) => Boolean(retenu && retenu.id === d.id && retenu.model === d.model);
  const unRetenu = candidats.some(estRetenu);

  return (
    <div className="verre mt-4 rounded-xl p-4">
      <p className="titre text-[0.9375rem] font-medium">Ce qu'on a entendu</p>
      {indices.resume ? (
        <p className="mt-1 text-ekn-sm text-fonce/70">{indices.resume}</p>
      ) : null}
      {entendu.length ? (
        <ul className="mt-2 flex flex-wrap gap-1.5">
          {entendu.map((terme) => (
            <li key={terme}
                className="rounded-full bg-papier px-2 py-0.5 text-ekn-xs text-fonce/70">
              {terme}
            </li>
          ))}
        </ul>
      ) : null}

      {candidats.length || etat.odoo ? (
        // Un bloc à part, sur fond clair : c'est une décision à prendre,
        // pas une information de plus. Chaque dossier est une carte avec
        // son bouton, et celui qui sera utilisé se voit d'un coup d'œil.
        <div className="mt-4 rounded-lg border border-turquoise/40 bg-white/80 p-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <p className="titre text-[0.9375rem] font-medium">Dossier Odoo de cette réunion</p>
            {enquete.record && estRetenu(enquete.record) ? (
              <span className="rounded-full bg-turquoise/20 px-2 py-0.5 text-ekn-xs font-medium text-fonce">
                {CERTITUDE[enquete.confidence] || 'Proposé'}
              </span>
            ) : null}
          </div>
          <p className="mt-1 text-ekn-sm text-fonce/60">
            {!candidats.length
              ? `Aucun dossier trouvé${enquete.reason ? ` : ${enquete.reason}` : '.'}`
              : enquete.record
                ? enquete.reason
                : 'Aucun n’est certain : choisis celui qui convient, son contexte guidera la transcription.'}
          </p>
          {candidats.length ? (
            <ul className="mt-3 space-y-2">
              {candidats.map((dossier) => (
                <Carte
                  key={`${dossier.model}-${dossier.id}`}
                  dossier={dossier}
                  choisi={estRetenu(dossier)}
                  surChoix={() => surChoix(dossier)}
                  surRetrait={surRetrait}
                />
              ))}
            </ul>
          ) : null}
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
            {unRetenu ? (
              <button
                type="button"
                onClick={surRetrait}
                className="text-ekn-sm text-ekn-text-muted underline-offset-2 hover:text-fonce hover:underline"
              >
                Aucun de ceux-là
              </button>
            ) : (
              <span className="text-ekn-sm text-ekn-text-muted">
                {candidats.length ? 'Aucun retenu : la transcription ne sera rattachée à rien.' : ''}
              </span>
            )}
            {etat.odoo ? (
              <button
                type="button"
                onClick={() => setDiscussion(true)}
                className="inline-flex items-center gap-1.5 rounded-full border border-bord bg-white px-3 py-1 text-ekn-sm text-fonce/70 transition-colors hover:border-fonce/40 hover:text-fonce"
              >
                <IconeBulle />
                {echanges.length ? 'Reprendre la recherche' : 'Pas le bon ? Guider la recherche'}
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      {(enquete.trace || []).length ? (
        <details className="mt-3">
          <summary className="cursor-pointer text-ekn-sm text-ekn-text-muted">
            Comment on est arrivé là
          </summary>
          <ol className="mt-1 space-y-0.5 text-ekn-sm text-ekn-text-muted">
            {enquete.trace.map((ligne, i) => (
              <li key={i}>· {ligne}</li>
            ))}
          </ol>
        </details>
      ) : null}

      {discussion ? (
        <Discussion
          indices={indices}
          moment={etat.moment}
          dejaVus={candidats}
          echanges={echanges}
          setEchanges={setEchanges}
          surChoix={(dossier) => {
            setDiscussion(false);
            surAjout(dossier);
          }}
          surFermer={() => setDiscussion(false)}
        />
      ) : null}
    </div>
  );
}

function Carte({ dossier, choisi, surChoix, surRetrait }) {
  return (
    <li
      className={`flex items-center gap-3 rounded-lg border p-2.5 transition-colors ${
        choisi ? 'border-turquoise-sombre bg-turquoise/10' : 'border-bord bg-white'
      }`}
    >
      <div className="min-w-0 flex-1">
        <p className="text-ekn-sm">
          <span className="titre font-medium">{dossier.name}</span>
          {dossier.partner ? <span className="text-ekn-text-muted"> · {dossier.partner}</span> : null}
        </p>
        <p className="text-ekn-sm text-ekn-text-muted">
          {[dossier.kind, dossier.reason,
            dossier.matched ? `trouvé sur « ${dossier.matched} »` : '',
            dossier.updated ? `modifié le ${dossier.updated}` : '']
            .filter(Boolean).join(' · ')}
        </p>
      </div>
      {choisi ? (
        // Recliquer défait : le geste qui coche est celui qui décoche,
        // sans chercher un bouton ailleurs.
        <button
          type="button"
          onClick={surRetrait}
          title="Ne plus utiliser ce dossier"
          className="group w-24 shrink-0 rounded-md px-3 py-1.5 text-ekn-sm font-medium text-ekn-success-dark hover:bg-fonce/5 hover:text-fonce"
        >
          <span className="group-hover:hidden">✓ utilisé</span>
          <span className="hidden group-hover:inline">Retirer</span>
        </button>
      ) : (
        <button
          type="button"
          onClick={surChoix}
          className="ekn-button ekn-button--compact w-24 shrink-0"
        >
          Utiliser
        </button>
      )}
    </li>
  );
}

function IconeBulle() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none"
         stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round">
      <path d="M2.5 3.5h11v7h-6l-3 2.5v-2.5h-2z" />
    </svg>
  );
}

/** Pendant que la sonde écoute et enquête.
 *
 *  Il faut voir que quelque chose se passe, sans en faire une attente :
 *  un point qui respire, ce qu'on est en train de faire, le temps
 *  écoulé, et la place que prendront les propositions.
 */
function Attente({ phase }) {
  const [secondes, setSecondes] = useState(0);
  useEffect(() => {
    const minuteur = setInterval(() => setSecondes((n) => n + 1), 1000);
    return () => clearInterval(minuteur);
  }, []);

  return (
    <div className="verre mt-4 rounded-xl p-4" aria-live="polite">
      <div className="flex items-center gap-2.5">
        <span className="relative flex h-2.5 w-2.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-turquoise opacity-60" />
          <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-turquoise-sombre" />
        </span>
        <p className="titre text-[0.9375rem] font-medium">Recherche du dossier Odoo</p>
        <span className="ml-auto text-ekn-xs tabular-nums text-ekn-text-muted">{secondes} s</span>
      </div>
      <p className="mt-1 text-ekn-sm text-fonce/60">
        {phase === 'recherche'
          ? 'Écoute des premières minutes, puis recherche dans ton agenda et tes dossiers…'
          : 'Préparation d’un extrait des premières minutes…'}
      </p>
      <div className="mt-3 space-y-2">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="h-11 animate-pulse rounded-lg bg-fonce/[0.06]"
            style={{ animationDelay: `${i * 180}ms` }}
          />
        ))}
      </div>
      <p className="mt-2 text-ekn-sm text-ekn-text-muted">
        Souvent moins d'une minute. Tu peux remplir la suite en attendant.
      </p>
    </div>
  );
}

/** Guider la recherche quand aucune proposition ne convient.
 *
 *  La personne connaît ses dossiers mieux que les indices : elle dit où
 *  chercher, l'enquête repart avec ses mots, et répond par une phrase
 *  et des dossiers. En choisir un le range dans la liste, retenu.
 */
function Discussion({ indices, moment, dejaVus, echanges, setEchanges, surChoix, surFermer }) {
  const [saisie, setSaisie] = useState('');
  const [attente, setAttente] = useState(false);
  const [erreur, setErreur] = useState('');
  const fil = useRef(null);
  const champ = useRef(null);

  useEffect(() => {
    champ.current?.focus();
    const echap = (e) => { if (e.key === 'Escape') surFermer(); };
    window.addEventListener('keydown', echap);
    return () => window.removeEventListener('keydown', echap);
  }, [surFermer]);

  useEffect(() => {
    fil.current?.scrollTo({ top: fil.current.scrollHeight, behavior: 'smooth' });
  }, [echanges, attente]);

  async function envoyer() {
    const texte = saisie.trim();
    if (!texte || attente) return;
    const suite = [...echanges, { role: 'user', texte }];
    setEchanges(suite);
    setSaisie('');
    setErreur('');
    setAttente(true);
    // Ce qu'on a déjà montré sans que ce soit choisi vaut refus : le
    // reproposer ferait tourner la conversation en rond.
    const montres = [...dejaVus, ...echanges.flatMap((e) => e.propositions || [])];
    const ecartes = [...new Map(montres.map((d) => [`${d.model}-${d.id}`, d])).values()]
      .map((d) => ({ model: d.model, id: d.id, name: d.name || '' }));
    try {
      const vue = await api.enqueteGuidee({
        clues: indices,
        moment: moment || '',
        rejected: ecartes,
        messages: suite.map(({ role, texte: t }) => ({ role, text: t })),
      });
      setEchanges([...suite, {
        role: 'assistant',
        texte: vue.answer || (vue.candidates.length ? 'Voici ce que j’ai trouvé.' : 'Rien trouvé.'),
        propositions: vue.candidates,
      }]);
    } catch (e) {
      setErreur(e.message);
    } finally {
      setAttente(false);
      champ.current?.focus();
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-fonce/30 p-4 backdrop-blur-[2px] sm:items-center"
      onMouseDown={(e) => { if (e.target === e.currentTarget) surFermer(); }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Guider la recherche du dossier"
        className="flex max-h-[85vh] w-full max-w-lg flex-col rounded-2xl bg-white shadow-2xl"
      >
        <div className="flex items-start justify-between gap-3 border-b border-bord px-5 py-4">
          <div>
            <p className="titre text-[1.0625rem] font-medium">Guider la recherche</p>
            <p className="mt-0.5 text-ekn-sm text-ekn-text-muted">
              Dis où chercher : le client, le type de dossier, une personne, un
              détail qui le distingue. Chaque message relance une recherche dans Odoo.
            </p>
          </div>
          <button
            type="button"
            onClick={surFermer}
            aria-label="Fermer"
            className="-mr-1 rounded-md px-2 text-[1.25rem] leading-none text-ekn-text-muted hover:bg-papier hover:text-fonce"
          >
            ×
          </button>
        </div>

        <div ref={fil} className="min-h-[8rem] flex-1 space-y-3 overflow-y-auto px-5 py-4">
          {echanges.map((e, i) => (e.role === 'user' ? (
            <p key={i} className="ml-auto w-fit max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-fonce px-3 py-2 text-ekn-sm text-clair">
              {e.texte}
            </p>
          ) : (
            <div key={i} className="max-w-[92%]">
              <p className="w-fit whitespace-pre-wrap rounded-2xl rounded-bl-md bg-papier px-3 py-2 text-ekn-sm text-fonce/85">
                {e.texte}
              </p>
              {(e.propositions || []).length ? (
                <ul className="mt-2 space-y-1.5">
                  {e.propositions.map((d) => (
                    <Carte
                      key={`${d.model}-${d.id}`}
                      dossier={d}
                      choisi={false}
                      surChoix={() => surChoix(d)}
                    />
                  ))}
                </ul>
              ) : null}
            </div>
          )))}
          {attente ? (
            <div className="flex w-fit items-center gap-2 rounded-2xl rounded-bl-md bg-papier px-3 py-2 text-ekn-sm text-ekn-text-muted">
              <span className="flex gap-1">
                {[0, 1, 2].map((i) => (
                  <span key={i} className="h-1.5 w-1.5 animate-bounce rounded-full bg-fonce/40"
                        style={{ animationDelay: `${i * 150}ms` }} />
                ))}
              </span>
              Recherche dans Odoo…
            </div>
          ) : null}
          {!echanges.length && !attente ? (
            <p className="text-ekn-sm text-ekn-text-muted">
              Les propositions déjà faites ne reviendront pas.
            </p>
          ) : null}
          {erreur ? <p role="alert" className="ekn-error">{erreur}</p> : null}
        </div>

        <div className="border-t border-bord px-5 py-3">
          <div className="flex items-end gap-2">
            <textarea
              ref={champ}
              rows={2}
              value={saisie}
              onChange={(e) => setSaisie(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); envoyer(); }
              }}
              placeholder="Ex. : ce n’est pas l’opportunité, c’est le projet de déploiement — le contact côté client est Mme Martin."
              className="min-h-[2.75rem] flex-1 resize-none rounded-lg border border-bord px-3 py-2 text-ekn-sm placeholder:text-ekn-text-muted focus:border-ekn-border-control"
            />
            <button
              type="button"
              onClick={envoyer}
              disabled={!saisie.trim() || attente}
              className="ekn-button ekn-button--compact"
            >
              Envoyer
            </button>
          </div>
          <p className="mt-1.5 text-ekn-sm text-ekn-text-muted">
            Entrée pour envoyer · Maj+Entrée pour aller à la ligne
          </p>
        </div>
      </div>
    </div>
  );
}
