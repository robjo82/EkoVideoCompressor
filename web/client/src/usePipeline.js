import { useCallback, useEffect, useState } from 'react';
import { api } from './api.js';

/** Pilote les transcriptions : création, encodage, envoi, suivi.
 *
 *  Les tâches vivent **hors des composants**. Le navigateur n'a qu'un
 *  rôle — encoder les fenêtres et les envoyer — et il le tient même quand
 *  on a quitté l'écran « Nouvelle transcription ». Une fois les fenêtres
 *  parties, le serveur finit seul : on peut revenir à la bibliothèque,
 *  lancer une deuxième transcription, voire fermer l'onglet.
 */

const taches = new Map(); // cle → état d'une transcription
const abonnes = new Set();

function publier(cle, champs) {
  taches.set(cle, { ...(taches.get(cle) || {}), ...champs });
  abonnes.forEach((f) => f());
}

function majFenetre(cle, index, champs) {
  const tache = taches.get(cle);
  if (!tache) return;
  publier(cle, {
    fenetres: tache.fenetres.map((f) => (f.index === index ? { ...f, ...champs } : f)),
  });
}

/** Encode ou envoie encore depuis cet onglet : le fermer tuerait le travail. */
export function encodeEncore() {
  return [...taches.values()].some(
    (t) => t.etat === 'creation' || (t.etat === 'traitement' && !t.envoiTermine),
  );
}

if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', (evenement) => {
    if (encodeEncore()) {
      evenement.preventDefault();
      evenement.returnValue = '';
    }
  });
}

function suivre(cle, jobId) {
  const tour = async () => {
    try {
      const vue = await api.job(jobId);
      const tache = taches.get(cle);
      if (!tache) return;
      publier(cle, {
        fenetres: tache.fenetres.map((f) => {
          const distante = vue.chunks.find((c) => c.index === f.index);
          if (!distante) return f;
          if (distante.status === 'termine') return { ...f, etat: 'terminee' };
          if (distante.status === 'erreur') return { ...f, etat: 'erreur', erreur: distante.error };
          return f;
        }),
      });
      if (vue.status === 'cancelled' || tache.etat === 'interrompu') {
        publier(cle, { etat: 'interrompu', message: '' });
        return;
      }
      if (vue.status === 'erreur') {
        publier(cle, { etat: 'repos', erreur: vue.error || 'La transcription a échoué.' });
        return;
      }
      if (vue.status === 'termine') {
        publier(cle, { etat: 'termine', message: '', resultat: { job_id: jobId, title: vue.title } });
        return;
      }
      if (vue.status === 'finalisation') publier(cle, { message: 'Fusion des fenêtres…' });
    } catch (e) {
      publier(cle, { erreur: e.message });
    }
    setTimeout(tour, 3000);
  };
  setTimeout(tour, 3000);
}

// Les workers en cours, par transcription : « Interrompre » doit pouvoir
// arrêter l'encodage sur-le-champ, pas au prochain envoi.
const workers = new Map();
// Ce qu'il faut pour reprendre une transcription au même endroit : le
// fichier reste en mémoire tant que l'onglet est ouvert.
const reprises = new Map();

async function demarrer(cle, { fichier, duree, contexte, modele, offset = 0, meetingDate }) {
  publier(cle, {
    etat: 'creation', message: 'Création du traitement…', erreur: '',
    fichier: fichier.name, fenetres: [], resultat: null, estimation: null, jobId: null,
  });
  let job;
  try {
    job = await api.createJob({
      filename: fichier.name,
      duration_seconds: duree,
      model: modele,
      language: 'fr',
      context: contexte,
      meeting_date: meetingDate || null,
      // Une réunion récupérée se transcrit sur place, sous le même numéro.
      reprocess_job_id: fichier.remote?.jobId ?? null,
    });
  } catch (e) {
    // Le garde-fou budget répond ici, avant le premier octet envoyé.
    publier(cle, { etat: 'repos', erreur: e.message });
    return;
  }

  publier(cle, {
    jobId: job.job_id,
    estimation: job.estimated_cost_usd,
    fenetres: job.chunks.map((c) => ({ ...c, etat: 'attendue', octets: 0 })),
    etat: 'traitement',
  });
  reprises.set(cle, { fichier, job, offset });
  await envoyer(cle);
  suivre(cle, job.job_id);
}

/** Encode et envoie ce qui manque encore au serveur — tout, au premier
 *  passage ; seulement les fenêtres absentes, après une erreur. */
async function envoyer(cle) {
  const { fichier, job, offset } = reprises.get(cle);
  publier(cle, { erreur: '', envoiTermine: false, message: 'Encodage sur ce poste…' });

  // Reprise : le serveur dit ce qui manque, on ne réencode que cela.
  const restant = (await api.job(job.job_id)).missing_chunks;
  // Interrompue pendant qu'on interrogeait le serveur : rien à encoder.
  if (taches.get(cle)?.etat === 'interrompu') return;
  const worker = new Worker(new URL('./media-worker.js', import.meta.url), { type: 'module' });
  workers.set(cle, worker);
  const finir = () => { worker.terminate(); workers.delete(cle); };
  worker.onmessage = ({ data }) => {
    if (data.kind === 'encoding') {
      majFenetre(cle, data.index, { etat: 'encodage', progression: data.ratio });
    } else if (data.kind === 'encoded') {
      majFenetre(cle, data.index, { etat: 'envoi', octets: data.bytes });
    } else if (data.kind === 'retrying') {
      publier(cle, {
        message: `Connexion perdue : nouvel essai dans ${Math.round(data.wait / 1000)} s (essai ${data.attempt + 1})…`,
      });
    } else if (data.kind === 'uploaded') {
      majFenetre(cle, data.index, { etat: 'transcription' });
      publier(cle, { message: 'Encodage sur ce poste…' });
    } else if (data.kind === 'done') {
      finir();
      publier(cle, {
        envoiTermine: true,
        message: 'Tout est envoyé : le serveur termine seul. Tu peux quitter cet écran.',
      });
    } else if (data.kind === 'cancelled') {
      finir();
      publier(cle, { etat: 'interrompu', message: '', erreur: '' });
    } else if (data.kind === 'error') {
      finir();
      publier(cle, { erreur: data.message, reprenable: true });
      // L'erreur part aussi au serveur : sans elle, on ne saurait d'un
      // « network error » que ce qu'en dit la personne qui l'a eu.
      api.reportClientError({
        job_id: job.job_id, stage: data.stage || '', file_name: fichier.name, message: data.message,
      }).catch(() => {});
    }
  };
  worker.postMessage({
    file: fichier, jobId: job.job_id, chunks: job.chunks, audio: job.audio,
    pending: restant, offset,
  });
}

/** Reprend une transcription en erreur au même endroit, sans recréer de
 *  réunion ni repayer les fenêtres déjà transcrites. */
export function reprendre(cle) {
  if (reprises.has(cle)) envoyer(cle);
}

/** Interrompt une transcription : l'encodage s'arrête tout de suite, le
 *  serveur ne transcrit plus rien, et la réunion part à la corbeille. */
export async function interrompre(cle) {
  const tache = taches.get(cle);
  workers.get(cle)?.terminate();
  workers.delete(cle);
  reprises.delete(cle);
  publier(cle, { etat: 'interrompu', message: '', erreur: '' });
  if (tache?.jobId) await api.cancelJob(tache.jobId).catch(() => {});
}

/** Interrompt une réunion depuis la bibliothèque. Si elle s'encode dans
 *  cet onglet, l'encodage s'arrête aussitôt ; sinon, le serveur refusera
 *  les envois du navigateur qui l'encode ailleurs. */
export async function interrompreReunion(jobId) {
  const cle = [...taches.entries()].find(([, t]) => t.jobId === jobId)?.[0];
  if (cle) return interrompre(cle);
  return api.cancelJob(jobId);
}

/** Toutes les transcriptions lancées depuis cet onglet. */
export function useTranscriptions() {
  const [, rafraichir] = useState(0);
  useEffect(() => {
    const f = () => rafraichir((n) => n + 1);
    abonnes.add(f);
    return () => abonnes.delete(f);
  }, []);
  return [...taches.values()];
}

/** L'écran de lancement ne suit qu'une transcription à la fois — la
 *  sienne. Les autres continuent, visibles dans la bibliothèque. */
export function usePipeline() {
  const [cle, setCle] = useState(null);
  const [, rafraichir] = useState(0);
  useEffect(() => {
    const f = () => rafraichir((n) => n + 1);
    abonnes.add(f);
    return () => abonnes.delete(f);
  }, []);

  const lancer = useCallback((options) => {
    const nouvelle = `t-${Date.now()}`;
    setCle(nouvelle);
    demarrer(nouvelle, options);
  }, []);

  /** Oublie la transcription suivie, sans l'arrêter : elle continue. */
  const detacher = useCallback(() => setCle(null), []);

  const tache = (cle && taches.get(cle)) || {};
  return {
    cle,
    reprenable: Boolean(tache.reprenable && tache.erreur),
    jobId: tache.jobId ?? null,
    fenetres: tache.fenetres || [],
    etat: tache.etat || 'repos',
    message: tache.message || '',
    erreur: tache.erreur || '',
    estimation: tache.estimation ?? null,
    resultat: tache.resultat || null,
    envoiTermine: Boolean(tache.envoiTermine),
    lancer,
    detacher,
  };
}
