import { useEffect, useState } from 'react';

/** Les fichiers déposés en attendant d'être lancés.
 *
 *  Déposer dix enregistrements d'un coup ne doit pas obliger à tout
 *  régler d'un coup : le premier s'ouvre dans le formulaire, les autres
 *  attendent ici, visibles dans la bibliothèque, et passent un à un.
 *
 *  La file vit dans l'onglet, hors des composants — comme les
 *  transcriptions en cours. Les fichiers restent sur le poste : fermer
 *  l'onglet vide la liste, sans rien perdre d'autre.
 */

const MEDIA_EXTENSIONS = /\.(m4a|mp3|wav|aac|flac|ogg|opus|weba|mp4|m4v|mov|mkv|webm|avi)$/i;

let queue = [];
let launchRequest = null;
const subscribers = new Set();

function publish(next) {
  queue = next;
  subscribers.forEach((notify) => notify());
}

/** Un fichier audio ou vidéo — le navigateur ne donne pas toujours le
 *  type, l'extension sert de repli. */
export function isMedia(file) {
  return /^(audio|video)\//.test(file.type || '') || MEDIA_EXTENSIONS.test(file.name || '');
}

/** Une réunion à retraiter, comme un fichier de la file.
 *
 *  Sa vidéo est déjà en stockage froid : le navigateur la relit de là,
 *  par plages, au lieu de demander le fichier d'origine — souvent jeté
 *  depuis. L'objet a le nom, le poids et la date d'un `File`, et dit en
 *  plus où lire, et quelle réunion il retraite. */
export function remoteFile(job) {
  const date = Date.parse(String(job.meeting_date || '').replace(' ', 'T'));
  return {
    name: job.filename,
    size: job.video_bytes,
    type: '',
    // La date sert à distinguer deux réunions de même nom ; c'est
    // `meetingDate` qui date la réunion à l'écran de lancement.
    lastModified: Number.isNaN(date) ? 0 : date,
    remote: {
      jobId: job.job_id,
      url: `/api/jobs/${job.job_id}/video`,
      meetingDate: job.meeting_date || null,
    },
  };
}

/** Ajoute des fichiers, sans doublon : le même fichier déposé deux fois
 *  n'attend qu'une fois. */
export function enqueue(files, { front = false } = {}) {
  const known = new Set(queue.map((e) => `${e.file.name}|${e.file.size}|${e.file.lastModified}`));
  const added = files
    .filter((file) => !known.has(`${file.name}|${file.size}|${file.lastModified}`))
    .map((file) => ({ id: `${Date.now()}-${Math.random().toString(36).slice(2)}`, file }));
  if (added.length) publish(front ? [...added, ...queue] : [...queue, ...added]);
  return added.length;
}

export function removeFromQueue(id) {
  publish(queue.filter((e) => e.id !== id));
}

/** Retire et rend le prochain fichier — ou celui demandé. */
export function takeNext(id = null) {
  const entry = id ? queue.find((e) => e.id === id) : queue[0];
  if (!entry) return null;
  publish(queue.filter((e) => e !== entry));
  return entry.file;
}

/** « Lancer » depuis la bibliothèque : l'écran de lancement le reprend
 *  en s'ouvrant. */
export function requestLaunch(id) {
  launchRequest = id;
}

/** Ouvre tout de suite dans l'écran de lancement, devant la file. */
export function launch(file) {
  enqueue([file], { front: true });
  const entry = queue.find((e) => e.file.name === file.name && e.file.size === file.size
    && e.file.lastModified === file.lastModified);
  if (entry) requestLaunch(entry.id);
}

export function consumeLaunchRequest() {
  const id = launchRequest;
  launchRequest = null;
  return id ? takeNext(id) : null;
}

export function useFileQueue() {
  const [, refresh] = useState(0);
  useEffect(() => {
    const notify = () => refresh((n) => n + 1);
    subscribers.add(notify);
    return () => subscribers.delete(notify);
  }, []);
  return queue;
}

/** Un fichier est-il en train d'être glissé au-dessus de la page ? Les
 *  événements de glisser arrivent par paires entrée/sortie sur chaque
 *  élément traversé : on compte, plutôt que de clignoter. */
export function useFileDrag() {
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
    const enter = (e) => { if (hasFiles(e)) { depth += 1; setDragging(true); } };
    const leave = (e) => { if (hasFiles(e)) { depth = Math.max(0, depth - 1); if (!depth) setDragging(false); } };
    // Lâché à côté de la zone, le navigateur ouvrirait le fichier à la
    // place de l'application : on l'en empêche partout.
    const over = (e) => { if (hasFiles(e)) e.preventDefault(); };
    const drop = (e) => { if (hasFiles(e)) e.preventDefault(); depth = 0; setDragging(false); };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('dragover', over);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('dragover', over);
      window.removeEventListener('drop', drop);
    };
  }, []);
  return dragging;
}
