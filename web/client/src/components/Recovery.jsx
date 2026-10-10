import { useEffect, useMemo, useState } from 'react';
import { api } from '../api.js';
import { Erreur } from './Communs.jsx';
import { duree, fileSize, jour } from '../format.js';

/** Récupérer son historique.
 *
 *  Avant transcript, les réunions finissaient dans le Drive, dans Odoo ou
 *  sur le Mac. Cet écran en fait l'inventaire et les range ici, à leur
 *  date. Rien ne bouge sans être coché : le Drive mêle réunions,
 *  tutoriels et voix off, et c'est à la personne de trier.
 */

const CONNECTION_MESSAGES = {
  connected: ['ok', 'Google Drive connecté.'],
  refused: ['erreur', 'Connexion annulée : rien n’a été partagé avec transcript.'],
  expired: ['erreur', 'La connexion a pris trop de temps. Recommence.'],
  wrong_account: ['erreur', 'Ce n’était pas ton compte Google Ekonum : connecte celui de ton adresse transcript.'],
  failed: ['erreur', 'Google n’a pas pu finaliser la connexion. Réessaie dans un instant.'],
};

const KIND_FILTERS = [
  ['meeting', 'Réunions probables'],
  ['unsure', 'À trier'],
  ['other', 'Probablement pas des réunions'],
  ['done', 'Déjà dans transcript'],
];

const LOCATION_LABELS = {
  my_drive: 'Mon Drive',
  shared_drive: 'Drive partagé',
  shared_with_me: 'Partagé avec moi',
};

export function Recovery({ connection, onOpen }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState('');
  const message = CONNECTION_MESSAGES[connection];

  useEffect(() => {
    api.recoveryState().then(setState).catch((e) => setError(e.message));
  }, []);

  return (
    <section className="mx-auto max-w-6xl px-6 py-10">
      <h1 className="titre text-[1.5rem] font-semibold">Récupérer ton historique</h1>
      <p className="mt-2 max-w-2xl text-fonce/70">
        Les réunions enregistrées avant transcript dorment peut-être dans ton
        Drive. On les retrouve, tu choisis lesquelles ranger ici — à la date où
        elles ont eu lieu — et les originaux peuvent partir à la corbeille du
        Drive une fois copiés.
      </p>
      {message ? (
        <p className={`mt-4 text-ekn-sm ${message[0] === 'ok' ? 'text-ekn-success-dark' : 'text-ekn-error-dark'}`}>
          {message[1]}
        </p>
      ) : null}
      <Erreur>{error}</Erreur>

      <div className="mt-8 space-y-6">
        {state ? (
          <DriveSource
            google={state.google}
            storage={state.storage}
            onGoogle={(google) => setState({ ...state, google })}
            onOpen={onOpen}
          />
        ) : (
          <p className="text-ekn-text-muted">Chargement…</p>
        )}
        <UpcomingSource
          title="Odoo"
          text="Les vidéos jointes aux fiches et les transcriptions collées dans leur fil de discussion, avec ta clé Odoo."
        />
        <UpcomingSource
          title="Ton Mac"
          text="Un dossier à parcourir depuis Chrome, pour les enregistrements restés sur ton poste."
        />
      </div>
    </section>
  );
}

function UpcomingSource({ title, text }) {
  return (
    <div className="verre rounded-xl p-5 opacity-70">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="titre text-[1.0625rem] font-medium">{title}</h2>
        <span className="text-ekn-sm text-ekn-text-muted">bientôt</span>
      </div>
      <p className="mt-1 text-ekn-sm text-fonce/60">{text}</p>
    </div>
  );
}

function DriveSource({ google, storage, onGoogle, onOpen }) {
  const [items, setItems] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const load = () => {
    setLoading(true);
    setError('');
    api.driveInventory()
      .then((vue) => setItems(vue.items))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { if (google.connected) load(); }, [google.connected]);

  return (
    <div className="verre rounded-xl p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="titre text-[1.0625rem] font-medium">Google Drive</h2>
        {google.connected ? (
          <span className="text-ekn-sm text-ekn-text-muted">
            {google.email} ·{' '}
            <button
              type="button"
              className="underline-offset-2 hover:text-fonce hover:underline"
              onClick={async () => {
                await api.googleDisconnect();
                setItems(null);
                onGoogle({ ...google, connected: false, email: '' });
              }}
            >
              déconnecter
            </button>
          </span>
        ) : null}
      </div>

      {!google.available ? (
        <p className="mt-2 text-ekn-sm text-fonce/60">
          La connexion à Google n’est pas encore configurée sur ce serveur.
        </p>
      ) : !google.connected ? (
        <div className="mt-2">
          <p className="text-ekn-sm text-fonce/60">
            Ton Drive, les Drive partagés dont tu es membre, et ce qu’on t’a
            partagé. transcript ne voit que ce que tu vois, et ne déplace un
            fichier qu’en ton nom, quand tu le demandes.
          </p>
          <a
            href="/api/google/connect"
            className="ekn-button mt-3"
          >
            Connecter mon Google Drive
          </a>
        </div>
      ) : loading && !items ? (
        <p className="mt-3 text-ekn-sm text-ekn-text-muted">Inventaire du Drive… cela peut prendre une minute.</p>
      ) : (
        <>
          <Erreur>{error}</Erreur>
          {items ? (
            <Inventory items={items} storage={storage} onReload={load} onOpen={onOpen} />
          ) : null}
        </>
      )}
    </div>
  );
}

function Inventory({ items, storage, onReload, onOpen }) {
  const [filter, setFilter] = useState('meeting');
  const [selected, setSelected] = useState(() => new Set(
    items.filter((i) => (i.status === 'new' && i.kind === 'meeting') || attachable(i)).map((i) => i.id),
  ));
  const [trashOriginals, setTrashOriginals] = useState(true);
  const [results, setResults] = useState({}); // id → { state, message, jobId }
  const [running, setRunning] = useState(false);

  const done = (i) => i.status !== 'new' && i.status !== 'in_library';
  const groups = useMemo(() => ({
    meeting: items.filter((i) => !done(i) && i.kind === 'meeting'),
    unsure: items.filter((i) => !done(i) && i.kind === 'unsure'),
    other: items.filter((i) => !done(i) && i.kind === 'other'),
    done: items.filter(done),
  }), [items]);
  const visible = groups[filter];
  // Un échec reste coché : un second clic sur « Récupérer » le retente.
  const chosen = items.filter((i) => selected.has(i.id) && (!results[i.id] || results[i.id].state === 'error'));
  const totalSize = chosen.reduce((sum, i) => sum + (i.size || 0), 0);

  const toggle = (id) => setSelected((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const recover = async () => {
    setRunning(true);
    for (const item of chosen) {
      setResults((r) => ({ ...r, [item.id]: { state: 'running' } }));
      try {
        // Déjà transcrite : le fichier rejoint sa réunion, il n'en crée pas
        // une seconde.
        const vue = attachable(item)
          ? await api.driveAttach(item.id, item.job_id, trashOriginals)
          : await api.driveImport(item.id, trashOriginals);
        setResults((r) => ({
          ...r,
          [item.id]: { state: 'done', jobId: vue.job_id, message: resultMessage(vue) },
        }));
      } catch (e) {
        setResults((r) => ({ ...r, [item.id]: { state: 'error', message: e.message } }));
      }
    }
    setRunning(false);
  };

  const recoveredCount = Object.values(results).filter((r) => r.state === 'done').length;

  if (!items.length) {
    return <p className="mt-3 text-ekn-sm text-fonce/60">Aucun fichier audio ou vidéo dans ce Drive.</p>;
  }

  return (
    <div className="mt-4">
      <div className="flex flex-wrap gap-1 text-ekn-sm">
        {KIND_FILTERS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setFilter(key)}
            className={`rounded-md px-3 py-1.5 transition-colors ${
              filter === key ? 'bg-white/70 font-medium' : 'text-ekn-text-muted hover:bg-white/40'
            }`}
          >
            {label} <span className="tabular-nums text-ekn-text-muted">{groups[key].length}</span>
          </button>
        ))}
        <button type="button" onClick={onReload} className="ml-auto px-2 text-ekn-sm text-ekn-text-muted hover:text-fonce">
          Relancer l’inventaire
        </button>
      </div>

      {visible.length ? (
        <div className="mt-3 overflow-x-auto rounded-lg border border-bord/70 bg-white/60">
          <table className="w-full min-w-[46rem] border-collapse text-ekn-sm">
            <thead className="border-b border-bord text-left text-ekn-sm text-ekn-text-muted">
              <tr>
                <th className="w-10 px-3 py-2" />
                <th className="px-3 py-2 font-medium">Fichier</th>
                <th className="px-3 py-2 font-medium">Date</th>
                <th className="px-3 py-2 font-medium">Durée</th>
                <th className="px-3 py-2 text-right font-medium">Taille</th>
                <th className="px-3 py-2 font-medium">Emplacement</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((item) => (
                <Row
                  key={item.id}
                  item={item}
                  checked={selected.has(item.id)}
                  result={results[item.id]}
                  disabled={running || done(item)}
                  onToggle={() => toggle(item.id)}
                  onOpen={onOpen}
                />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="mt-3 text-ekn-sm text-ekn-text-muted">Rien dans cette catégorie.</p>
      )}

      <div className="ekn-dark sticky bottom-4 mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl bg-fonce px-4 py-3 text-clair shadow-xl">
        <span className="text-ekn-sm tabular-nums">
          {running
            ? `Récupération… ${recoveredCount}/${recoveredCount + chosen.length}`
            : chosen.length
              ? `${chosen.length} fichier${chosen.length > 1 ? 's' : ''} · ${fileSize(totalSize)}`
              : recoveredCount
                ? `${recoveredCount} fichier${recoveredCount > 1 ? 's' : ''} rangé${recoveredCount > 1 ? 's' : ''} à leur date dans ta bibliothèque`
                : 'Coche les fichiers à récupérer'}
        </span>
        <label className="flex items-center gap-2 text-ekn-sm text-clair/80">
          <input
            type="checkbox"
            checked={trashOriginals}
            disabled={running}
            onChange={(e) => setTrashOriginals(e.target.checked)}
            className="h-4 w-4 accent-turquoise"
          />
          Mettre les originaux à la corbeille du Drive une fois copiés (30 jours pour se raviser)
        </label>
        <button
          type="button"
          disabled={running || !chosen.length || !storage}
          onClick={recover}
          title={storage ? '' : 'Le stockage vidéo n’est pas configuré'}
          className="ml-auto rounded-lg bg-turquoise px-4 py-1.5 text-ekn-sm font-medium text-fonce hover:bg-turquoise/90 disabled:opacity-40"
        >
          Récupérer
        </button>
      </div>
    </div>
  );
}

// Un fichier qui enregistre une réunion déjà à soi dans transcript : on le
// range avec elle plutôt que d'en créer une seconde.
const attachable = (item) => item.status === 'in_library' && Boolean(item.job_id);

function resultMessage(vue) {
  const rangement = vue.video === 'attached'
    ? 'rangée avec la réunion existante, qui garde désormais son enregistrement'
    : vue.video === 'kept' ? 'rangée avec la réunion existante' : 'récupérée';
  const original = {
    trashed: 'original à la corbeille du Drive',
    kept: 'original gardé (droits insuffisants pour le retirer)',
    kept_different: 'original gardé : il diffère de l’enregistrement déjà conservé',
  }[vue.original];
  return original ? `${rangement} · ${original}` : rangement;
}

function Row({ item, checked, result, disabled, onToggle, onOpen }) {
  const status = {
    recovered: item.recovered_by ? `déjà récupéré${item.job_id ? '' : ` par ${item.recovered_by}`}` : 'déjà récupéré',
    duplicate: 'copie d’un autre fichier de la liste',
    in_library: item.job_id
      ? 'déjà transcrite : sera rangée avec la réunion existante'
      : 'semble déjà transcrite par un collègue',
  }[item.status];

  return (
    <tr className="border-b border-bord/50 last:border-0">
      <td className="px-3 py-2 align-top">
        {result?.state === 'done' ? (
          <span className="text-ekn-success-dark">✓</span>
        ) : result?.state === 'running' ? (
          <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-fonce/20 border-t-fonce/70" />
        ) : (
          <input
            type="checkbox"
            checked={checked}
            disabled={disabled}
            onChange={onToggle}
            aria-label={`Récupérer ${item.name}`}
            className="mt-0.5 h-4 w-4 accent-turquoise-sombre disabled:opacity-30"
          />
        )}
      </td>
      <td className="px-3 py-2">
        <a href={item.web_link} target="_blank" rel="noreferrer" className="hover:underline">
          {item.name}
        </a>
        {status || result ? (
          <span className={`block text-ekn-sm ${result?.state === 'error' ? 'text-ekn-error-dark' : 'text-ekn-text-muted'}`}>
            {result?.message || status}
            {(result?.jobId || item.job_id) ? (
              <button
                type="button"
                onClick={() => onOpen(result?.jobId || item.job_id)}
                className="ml-2 text-violet hover:underline"
              >
                ouvrir
              </button>
            ) : null}
          </span>
        ) : null}
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-fonce/70">
        {jour(item.recorded_at)}
        {!item.recorded_at_from_name ? (
          <span className="block text-ekn-xs text-ekn-text-muted" title="Date du dépôt dans le Drive : l’enregistrement peut être un peu plus ancien.">
            date du Drive
          </span>
        ) : null}
      </td>
      <td className="px-3 py-2 tabular-nums text-fonce/70">
        {item.duration_seconds ? duree(item.duration_seconds) : '—'}
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-right tabular-nums text-fonce/70">{fileSize(item.size)}</td>
      <td className="px-3 py-2 text-fonce/60">
        {item.location.kind === 'shared_drive' ? item.location.name : LOCATION_LABELS[item.location.kind]}
      </td>
    </tr>
  );
}
