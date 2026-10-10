/**
 * Découpe et encode les fenêtres audio, hors du fil principal.
 *
 * C'est le seul endroit du projet qui touche au média, et il vit dans le
 * navigateur : le serveur n'a ni le stockage ni le CPU pour cela. Le
 * fichier source n'est jamais lu en entier — Mediabunny lit en flux, et
 * la mémoire ne suit donc pas la taille du fichier (119 Mo de pic mesurés
 * sur une source de 4,4 Go au jalon M0).
 *
 * L'envoi part d'ici aussi : repasser des ArrayBuffers de plusieurs Mo au
 * fil principal juste pour les poster n'apporterait que des copies.
 */
import {
  Input, Output, Conversion, ALL_FORMATS, BlobSource, UrlSource,
  BufferTarget, StreamTarget, Mp3OutputFormat, Mp4OutputFormat,
  OggOutputFormat, Quality,
} from 'mediabunny';
// WebCodecs n'encode que l'Opus et l'AAC : le MP3 vient d'un paquet
// d'extension Mediabunny, qui s'enregistre auprès du cœur. Il n'est pas
// embarqué d'office dans le paquet principal.
import { registerMp3Encoder } from '@mediabunny/mp3-encoder';

registerMp3Encoder();

const say = (message) => self.postMessage(message);

/** D'où lire le média : le fichier choisi sur ce poste, ou la vidéo déjà
 *  stockée d'une réunion qu'on retraite. Celle-ci se lit par plages, en
 *  flux comme un fichier local — plusieurs gigaoctets ne se téléchargent
 *  pas d'abord. L'en-tête dit au serveur que ce n'est pas une lecture. */
function sourceOf(file) {
  if (file?.remote) {
    return new UrlSource(file.remote.url, {
      requestInit: { headers: { 'X-Video-Purpose': 'processing' } },
    });
  }
  return new BlobSource(file);
}

function outputFormat(profile) {
  // Le conteneur suit le codec : le MP3 est nu, l'Opus a besoin d'un
  // conteneur. Ogg plutôt que WebM parce que c'est « audio/ogg » que
  // Gemini documente — reste à vérifier qu'il accepte l'Opus qui est
  // dedans, ce que le profil côté serveur tranchera le moment venu.
  if (profile.codec === 'mp3') return { format: new Mp3OutputFormat(), type: 'audio/mpeg' };
  return { format: new OggOutputFormat(), type: 'audio/ogg' };
}

self.onmessage = async (event) => {
  // Sonder la durée est la seule opération média dont le fil principal
  // aurait besoin : la faire ici lui évite d'embarquer Mediabunny, qui
  // pèse plus que tout le reste de l'interface réunie.
  if (event.data.kind === 'probe') {
    try {
      const input = new Input({ formats: ALL_FORMATS, source: sourceOf(event.data.file) });
      say({ kind: 'probed', duration: await input.computeDuration() });
    } catch (error) {
      say({ kind: 'probe-error', message: error?.message || String(error) });
    }
    return;
  }

  // Fenêtre isolée, rendue au fil principal plutôt qu'envoyée : la
  // sonde d'identification tourne avant qu'un traitement existe, donc
  // avant qu'il y ait une URL où pousser quoi que ce soit.
  if (event.data.kind === 'window') {
    try {
      const { file, start, end, audio } = event.data;
      const { format, type } = outputFormat(audio);
      const target = new BufferTarget();
      const conversion = await Conversion.init({
        input: new Input({ formats: ALL_FORMATS, source: sourceOf(file) }),
        output: new Output({ format, target }),
        trim: { start, end },
        video: { discard: true },
        audio: {
          codec: audio.codec,
          numberOfChannels: audio.channels,
          sampleRate: audio.sample_rate,
          quality: new Quality({ bitrate: audio.bitrate }),
        },
      });
      await conversion.execute();
      self.postMessage({ kind: 'window-done', bytes: target.buffer, type },
                       [target.buffer]);
    } catch (error) {
      say({ kind: 'window-error', message: error?.message || String(error) });
    }
    return;
  }

  if (event.data.kind === 'compress') {
    await compresser(event.data);
    return;
  }

  const { file, jobId, chunks, audio, pending, offset = 0 } = event.data;
  const todo = new Set(pending);
  let stage = 'encodage';

  try {
    for (const chunk of chunks) {
      if (!todo.has(chunk.index)) continue;   // déjà transcrite : on ne repaie pas

      stage = 'encodage';
      const started = performance.now();
      const { format, type } = outputFormat(audio);
      const target = new BufferTarget();
      try {
        const conversion = await Conversion.init({
          input: new Input({ formats: ALL_FORMATS, source: sourceOf(file) }),
          output: new Output({ format, target }),
          // Le plan de découpage est exprimé dans le temps *retenu* ;
          // l'offset le ramène sur la source quand l'utilisateur a rogné.
          trim: { start: offset + chunk.start, end: offset + chunk.end },
          video: { discard: true },
          audio: {
            codec: audio.codec,
            numberOfChannels: audio.channels,
            sampleRate: audio.sample_rate,
            quality: new Quality({ bitrate: audio.bitrate }),
          },
        });
        conversion.onProgress = (ratio) =>
          say({ kind: 'encoding', index: chunk.index, ratio });
        await conversion.execute();
      } catch (error) {
        // Chrome dit « network error » quand un fichier a changé sur le
        // disque depuis sa sélection (synchronisation iCloud, par
        // exemple) : ce n'est pas le réseau, c'est la lecture.
        throw new Error(
          `lecture du fichier impossible (fenêtre ${chunk.index + 1}) : ${error?.message || error}`,
        );
      }

      const bytes = target.buffer;
      say({ kind: 'encoded', index: chunk.index, bytes: bytes.byteLength,
            ms: performance.now() - started });

      stage = 'envoi';
      const response = await envoyer(`/api/jobs/${jobId}/chunks/${chunk.index}`, bytes, type,
        (attempt, wait) => say({ kind: 'retrying', index: chunk.index, attempt, wait }));
      if (response.status === 409) {
        say({ kind: 'cancelled' });
        return;
      }
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`envoi de la fenêtre ${chunk.index + 1} refusé (${response.status}) : ${detail}`);
      }
      say({ kind: 'uploaded', index: chunk.index });
    }
    say({ kind: 'done' });
  } catch (error) {
    say({ kind: 'error', stage, message: error?.message || String(error) });
  }
};

/** Envoie une fenêtre, en réessayant sur ce qui passe : coupure réseau,
 *  serveur qui redémarre (5xx), trop de demandes (429). Un refus franc
 *  (4xx) ne se réessaie pas — il dit quelque chose. */
async function envoyer(url, bytes, type, onRetry) {
  const attentes = [2000, 5000, 15000, 30000];
  for (let essai = 0; ; essai += 1) {
    try {
      const response = await fetch(url, {
        method: 'PUT', headers: { 'Content-Type': type }, body: bytes,
      });
      const passager = response.status >= 500 || response.status === 429;
      if (!passager || essai >= attentes.length) return response;
    } catch (error) {
      if (essai >= attentes.length) {
        throw new Error(`serveur injoignable après ${essai + 1} essais : ${error?.message || error}`);
      }
    }
    onRetry(essai + 1, attentes[essai]);
    await new Promise((resolve) => setTimeout(resolve, attentes[essai]));
  }
}

/** Compression, écrite directement sur le disque de l'utilisateur.
 *
 *  Le profil vient de la mesure M0 : HEVC 720p à 12 images par seconde,
 *  ~150 kbps, audio AAC 64 kbps mono. Le débit s'est révélé sans effet
 *  sur la lisibilité du texte à l'écran — l'encodeur plafonne vers
 *  133 kbps et la zone de texte est identique de 60 à 250 — donc rien ne
 *  justifie de dépenser plus.
 *
 *  La sortie passe par un `StreamTarget` vers un flux d'écriture : une
 *  archive de plusieurs centaines de mégaoctets ne tient pas en mémoire,
 *  et surtout **elle n'est jamais envoyée au serveur**.
 */
async function compresser({ file, handle, profile, trim }) {
  const debut = performance.now();
  try {
    const writable = await handle.createWritable();
    const conversion = await Conversion.init({
      input: new Input({ formats: ALL_FORMATS, source: sourceOf(file) }),
      output: new Output({
        format: new Mp4OutputFormat(),
        target: new StreamTarget(writable),
      }),
      // Si l'utilisateur a rogné, l'archive suit : garder ce qu'on a
      // décidé de ne pas transcrire n'aurait pas de sens.
      ...(trim ? { trim } : {}),
      video: {
        height: profile.height,
        fit: 'contain',
        codec: profile.codec,
        frameRate: profile.frameRate,
        quality: new Quality({ bitrate: profile.videoBitrate }),
      },
      audio: {
        codec: 'aac',
        numberOfChannels: 1,
        quality: new Quality({ bitrate: profile.audioBitrate }),
      },
    });
    conversion.onProgress = (ratio) => say({ kind: 'compress-progress', ratio });
    await conversion.execute();

    const ecrit = await handle.getFile();
    say({
      kind: 'compressed',
      bytes: ecrit.size,
      ms: performance.now() - debut,
    });
  } catch (error) {
    say({ kind: 'compress-error', message: error?.message || String(error) });
  }
}
