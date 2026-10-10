import { useEffect, useState } from 'react';
import { PROFIL } from './useCompression.js';

/** Archivage de la vidéo compressée : compresser, puis envoyer.
 *
 *  Vit hors des composants, exprès. Une réunion d'une heure se transcrit
 *  en deux minutes mais se compresse en douze : on a quitté l'écran
 *  « Nouvelle transcription » bien avant la fin. Lié à un composant, le
 *  travail mourrait avec lui.
 *
 *  La vidéo compressée passe par l'espace privé du navigateur (OPFS),
 *  pas par un « Enregistrer sous » : elle n'a rien à faire sur le disque
 *  de la personne, sa place est dans le stockage froid. Le fichier
 *  temporaire est effacé dès l'envoi terminé.
 */

const taches = new Map(); // jobId → { etape, progression, erreur }
const abonnes = new Set();

function publier(jobId, champs) {
  taches.set(jobId, { ...(taches.get(jobId) || {}), ...champs });
  abonnes.forEach((f) => f());
}

function enCours() {
  return [...taches.values()].some((t) => t.etape === 'compression' || t.etape === 'envoi');
}

// Fermer l'onglet tue la compression et l'envoi : on prévient, sans
// bloquer — c'est le navigateur qui affiche la question.
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', (evenement) => {
    if (enCours()) {
      evenement.preventDefault();
      evenement.returnValue = '';
    }
  });
}

/** Suivre l'archivage d'une réunion depuis n'importe quel écran. */
export function useArchivage(jobId) {
  const [, rafraichir] = useState(0);
  useEffect(() => {
    const f = () => rafraichir((n) => n + 1);
    abonnes.add(f);
    return () => abonnes.delete(f);
  }, []);
  return jobId == null ? null : taches.get(jobId) || null;
}

async function compresserVers(fichier, trim, poignee, surProgression) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./media-worker.js', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = ({ data }) => {
      if (data.kind === 'compress-progress') surProgression(data.ratio);
      else if (data.kind === 'compressed') { worker.terminate(); resolve(data.bytes); }
      else if (data.kind === 'compress-error') {
        worker.terminate();
        reject(new Error(data.message));
      }
    };
    worker.onerror = (e) => { worker.terminate(); reject(new Error(e.message)); };
    worker.postMessage({ kind: 'compress', file: fichier, handle: poignee, profile: PROFIL, trim });
  });
}

async function envoyer(jobId, blob, surProgression, remplacer) {
  const ouverture = await fetch(`/api/jobs/${jobId}/video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taille: blob.size, type: 'video/mp4', remplacer }),
  });
  if (!ouverture.ok) throw new Error((await ouverture.json()).detail || 'Envoi refusé.');
  const { morceau } = await ouverture.json();

  for (let debut = 0; debut < blob.size; debut += morceau) {
    const tranche = blob.slice(debut, Math.min(debut + morceau, blob.size));
    // Un morceau raté se retente : sur une heure de vidéo, un hoquet
    // réseau est probable, et tout recommencer serait absurde.
    let essai = 0;
    for (;;) {
      const reponse = await fetch(`/api/jobs/${jobId}/video?debut=${debut}`, {
        method: 'PUT', body: tranche,
      }).catch(() => null);
      if (reponse?.ok) break;
      if (++essai >= 4) {
        throw new Error('Envoi de la vidéo interrompu — le réseau a lâché trop longtemps.');
      }
      await new Promise((r) => setTimeout(r, 2000 * essai));
    }
    surProgression(Math.min(1, (debut + morceau) / blob.size));
  }
}

/** Compresse puis archive la vidéo d'une réunion.
 *
 *  `cible.jobId` peut n'être connu qu'après coup : la compression démarre
 *  aussitôt, le traitement est créé en parallèle, et l'envoi attend les
 *  deux.
 */
export async function archiver({ fichier, trim = null, cible }) {
  const cle = cible.cle;
  publier(cle, { etape: 'compression', progression: 0, erreur: '' });
  const racine = await navigator.storage.getDirectory();
  const nom = `archive-${Date.now()}.mp4`;
  try {
    const poignee = await racine.getFileHandle(nom, { create: true });
    await compresserVers(fichier, trim, poignee, (p) => publier(cle, { progression: p }));

    while (cible.jobId == null) {
      if (cible.abandon) throw new Error('Traitement abandonné.');
      await new Promise((r) => setTimeout(r, 500));
    }
    // L'état suit désormais la réunion, pour que sa fiche l'affiche.
    taches.delete(cle);
    publier(cible.jobId, { etape: 'envoi', progression: 0, erreur: '' });
    const compresse = await poignee.getFile();
    // Retraiter une réunion récupérée : sa vidéo stockée n'est remplacée
    // que si la nouvelle est vraiment plus légère.
    if (fichier.remote && compresse.size >= fichier.size) {
      publier(cible.jobId, {
        etape: 'termine', progression: 1,
        note: 'La vidéo stockée était déjà plus légère : gardée telle quelle.',
      });
      return;
    }
    await envoyer(
      cible.jobId, compresse, (p) => publier(cible.jobId, { progression: p }),
      Boolean(fichier.remote),
    );
    publier(cible.jobId, { etape: 'termine', progression: 1, octets: compresse.size });
  } catch (erreur) {
    publier(cible.jobId ?? cle, { etape: 'erreur', erreur: erreur.message });
  } finally {
    await racine.removeEntry(nom).catch(() => {});
  }
}
