import { useEffect, useRef, useState } from 'react';
import { horodatage } from '../format.js';

/** « 1:02:03 », « 12:30 » ou « 45 » → secondes ; null si illisible. */
function lireTemps(texte) {
  const parties = String(texte || '').trim().split(':').map((p) => p.trim());
  if (!parties.length || parties.some((p) => p === '' || Number.isNaN(Number(p)))) return null;
  return parties.reduce((total, p) => total * 60 + Number(p), 0);
}

/** Réécoute et rognage avant de lancer.
 *
 *  Deux besoins que l'app macOS couvrait : vérifier qu'on a le bon
 *  enregistrement, et couper le bavardage du début ou la demi-heure
 *  oubliée à la fin. Transcrire ce qu'on va jeter coûte de l'argent et
 *  pollue le transcript.
 *
 *  Le geste principal est la frise : la partie gardée est surlignée, et
 *  ses deux bords s'attrapent à la souris. Un seul bouton de lecture, qui
 *  part de la dernière poignée touchée — c'est là qu'on veut entendre si
 *  la coupe tombe juste. Pas de lecteur natif : il dépareille, et ses
 *  contrôles font double emploi avec la frise.
 *
 *  Le fichier se lit par une URL d'objet : rien ne part au serveur.
 */
export function Apercu({ fichier, duree, debut, fin, surDebut, surFin, actif }) {
  const [url, setUrl] = useState('');
  const [position, setPosition] = useState(0);
  const [lisible, setLisible] = useState(true);
  const [image, setImage] = useState(false); // a-t-il une piste vidéo ?
  const [joue, setJoue] = useState(false);
  const [dernier, setDernier] = useState('debut'); // dernière poignée touchée
  const media = useRef(null);
  const frise = useRef(null);
  const [prise, setPrise] = useState(null); // 'debut' | 'fin' | null

  useEffect(() => {
    if (!fichier) return undefined;
    setLisible(true);
    setImage(false);
    setJoue(false);
    // Une réunion qu'on retraite se lit là où sa vidéo est stockée.
    if (fichier.remote) {
      setUrl(`${fichier.remote.url}?purpose=processing`);
      return undefined;
    }
    const objet = URL.createObjectURL(fichier);
    setUrl(objet);
    // Sans révocation, chaque changement de fichier laisse le précédent
    // épinglé en mémoire par le navigateur.
    return () => URL.revokeObjectURL(objet);
  }, [fichier]);

  if (!fichier || !duree) return null;

  const rogne = debut > 0 || fin < duree;
  const pct = (t) => `${(Math.min(Math.max(t, 0), duree) / duree) * 100}%`;

  const tempsSous = (clientX) => {
    const cadre = frise.current.getBoundingClientRect();
    const ratio = Math.min(Math.max((clientX - cadre.left) / cadre.width, 0), 1);
    return ratio * duree;
  };

  const aller = (t) => {
    if (media.current) media.current.currentTime = t;
    setPosition(t);
  };

  const placer = (bord, t) => {
    setDernier(bord);
    // Une seconde d'écart au moins : une plage vide ne se transcrit pas.
    if (bord === 'debut') surDebut(Math.min(Math.max(t, 0), fin - 1));
    else surFin(Math.max(Math.min(t, duree), debut + 1));
  };

  const basculer = () => {
    const lecteur = media.current;
    if (!lecteur) return;
    if (!lecteur.paused) { lecteur.pause(); return; }
    // Depuis le début : on écoute ce qui suit la coupe. Depuis la fin :
    // on écoute les quelques secondes qui la précèdent.
    lecteur.currentTime = dernier === 'fin' ? Math.max(fin - 5, debut) : debut;
    lecteur.play().catch(() => setLisible(false));
  };

  const poignee = (bord, valeur) => (
    <button
      type="button"
      disabled={actif}
      aria-label={bord === 'debut' ? 'Début de la partie gardée' : 'Fin de la partie gardée'}
      onPointerDown={(e) => {
        e.stopPropagation();
        e.currentTarget.setPointerCapture(e.pointerId);
        setPrise(bord);
        setDernier(bord);
      }}
      onPointerMove={(e) => {
        if (prise !== bord) return;
        const t = tempsSous(e.clientX);
        placer(bord, t);
        aller(t);
      }}
      onPointerUp={() => setPrise(null)}
      onKeyDown={(e) => {
        const pas = e.shiftKey ? 10 : 1;
        if (e.key === 'ArrowLeft') placer(bord, valeur - pas);
        if (e.key === 'ArrowRight') placer(bord, valeur + pas);
      }}
      style={{ left: pct(valeur) }}
      className={`absolute top-1/2 z-10 h-7 w-4 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize rounded-md border-2 border-white shadow focus:outline-none focus:ring-2 focus:ring-turquoise disabled:cursor-not-allowed ${
        dernier === bord ? 'bg-fonce' : 'bg-turquoise-sombre'
      } ${prise === bord ? 'scale-110' : ''}`}
    />
  );

  const champTemps = (libelle, valeur, bord) => (
    <label className="flex items-center gap-2 text-[0.8125rem] text-fonce/60">
      {libelle}
      <input
        key={`${bord}-${Math.round(valeur)}`}
        defaultValue={horodatage(valeur)}
        disabled={actif}
        onFocus={() => setDernier(bord)}
        onBlur={(e) => {
          const t = lireTemps(e.target.value);
          if (t === null) e.target.value = horodatage(valeur);
          else placer(bord, t);
        }}
        onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); }}
        className="w-20 rounded-md border border-bord bg-white px-2 py-1 text-center tabular-nums text-fonce"
      />
    </label>
  );

  return (
    <div className="verre mt-4 rounded-xl p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="titre text-[0.9375rem] font-medium">Écouter et rogner</p>
        {/* « tout reprendre » vit ici, dans l'en-tête : placé entre les
            champs, il faisait passer la ligne à la ligne et remonter la
            frise sous le curseur — au pire moment, en pleine prise. */}
        <p className="text-[0.8125rem] tabular-nums text-fonce/60">
          {rogne ? (
            <>
              <span className="text-turquoise-sombre">
                {horodatage(fin - debut)} gardées sur {horodatage(duree)}
              </span>
              <button
                type="button"
                disabled={actif}
                onClick={() => { surDebut(0); surFin(duree); }}
                className="ml-3 text-fonce/55 underline-offset-2 hover:text-fonce hover:underline"
              >
                tout reprendre
              </button>
            </>
          ) : `${horodatage(duree)} — tout est gardé`}
        </p>
      </div>

      <div className="mt-3 flex flex-col gap-4 sm:flex-row sm:items-center">
        {/* L'image n'apparaît que s'il y en a une : un fichier audio n'a
            pas à réserver un rectangle noir. */}
        <video
          ref={media}
          src={url}
          playsInline
          preload="metadata"
          onLoadedMetadata={(e) => setImage(e.currentTarget.videoWidth > 0)}
          onTimeUpdate={(e) => {
            const t = e.currentTarget.currentTime;
            setPosition(t);
            // La lecture s'arrête au bord de la partie gardée : au-delà,
            // ce qu'on entend ne sera pas transcrit.
            if (t >= fin) e.currentTarget.pause();
          }}
          onPlay={() => setJoue(true)}
          onPause={() => setJoue(false)}
          onError={() => setLisible(false)}
          className={image ? 'aspect-video w-full shrink-0 rounded-lg bg-fonce sm:w-48' : 'hidden'}
        />

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={basculer}
              disabled={!lisible}
              title={lisible
                ? `Écouter depuis ${dernier === 'fin' ? 'la fin' : 'le début'} de la partie gardée`
                : 'Format non lisible par ce navigateur'}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-fonce text-clair transition-colors hover:bg-fonce-doux disabled:opacity-30"
            >
              {joue ? (
                <span className="flex gap-1" aria-label="Pause">
                  <span className="h-3.5 w-1 rounded-sm bg-current" />
                  <span className="h-3.5 w-1 rounded-sm bg-current" />
                </span>
              ) : (
                <span
                  aria-label="Lecture"
                  className="ml-0.5 h-0 w-0 border-y-[7px] border-l-[11px] border-y-transparent border-l-current"
                />
              )}
            </button>

            {/* La frise : cliquer déplace la lecture, les poignées bornent
                la partie gardée. */}
            <div
              ref={frise}
              onPointerDown={(e) => aller(tempsSous(e.clientX))}
              className="relative h-10 flex-1 cursor-pointer select-none"
            >
              <div className="absolute inset-x-0 top-1/2 h-2 -translate-y-1/2 rounded-full bg-fonce/10" />
              <div
                className="absolute top-1/2 h-2 -translate-y-1/2 rounded-full bg-turquoise"
                style={{ left: pct(debut), width: `calc(${pct(fin)} - ${pct(debut)})` }}
              />
              <div
                className="pointer-events-none absolute top-1 bottom-1 w-px bg-fonce/70"
                style={{ left: pct(position) }}
              />
              {poignee('debut', debut)}
              {poignee('fin', fin)}
            </div>
          </div>

          <div className="mt-2 flex flex-wrap items-center justify-between gap-3 pl-12">
            {champTemps('Début', debut, 'debut')}
            <span className="text-[0.75rem] tabular-nums text-fonce/45">
              {joue || position > 0 ? horodatage(position) : ''}
            </span>
            {champTemps('Fin', fin, 'fin')}
          </div>
        </div>
      </div>
      <p className="mt-2 text-[0.75rem] text-fonce/45">
        {lisible
          ? 'Seule la partie surlignée sera transcrite — et payée. La lecture part de la dernière poignée touchée.'
          : 'Format non lisible par ce navigateur : le rognage reste possible, à l’aveugle.'}
      </p>
    </div>
  );
}
