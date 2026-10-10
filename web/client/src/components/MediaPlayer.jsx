import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';

/** <ekn-audio> d'ekonum-ui, monté hors de React.
 *
 *  Le composant dessine ses contrôles à côté de la <video> et, pour
 *  rogner, la déplace dans son cadre : React ne doit pas gérer ses
 *  enfants. On l'assemble donc à la main dans un hôte vide, et on le
 *  recrée quand la source change — `start` et `end` ne sont lus qu'au
 *  chargement.
 */
export const EknAudio = forwardRef(function EknAudio(
  { src, trim = false, start, end, label, autoPlay = false, onTrim, onTime, onUnreadable, className = '', inert },
  ref,
) {
  const hote = useRef(null);
  const lecteur = useRef(null);
  const rappels = useRef({});
  rappels.current = { onTrim, onTime, onUnreadable };

  useImperativeHandle(ref, () => ({
    seek: (seconds, play = true) => lecteur.current?.seek(seconds, play),
  }), []);

  useEffect(() => {
    if (!src) return undefined;
    const el = document.createElement('ekn-audio');
    if (trim) el.setAttribute('trim', '');
    if (label) el.setAttribute('label', label);
    if (start != null) el.setAttribute('start', String(start));
    if (end != null) el.setAttribute('end', String(end));
    const media = document.createElement('video');
    media.preload = 'metadata';
    media.playsInline = true;
    media.controls = true; // sans le script, le lecteur du navigateur reste
    media.autoplay = autoPlay;
    media.src = src;
    el.append(media);
    el.addEventListener('ekn-trim', (e) => rappels.current.onTrim?.(e.detail));
    media.addEventListener('timeupdate', () => rappels.current.onTime?.(media.currentTime));
    media.addEventListener('error', () => rappels.current.onUnreadable?.());
    hote.current.append(el);
    lecteur.current = el;
    return () => {
      // Libère la lecture et la connexion : retirer l'élément ne suffit pas.
      media.pause();
      media.removeAttribute('src');
      media.load();
      el.remove();
      lecteur.current = null;
    };
    // start / end ne valent qu'au chargement : les suivre recréerait le
    // lecteur à chaque poignée déplacée.
  }, [src, trim, label, autoPlay]);

  return <div ref={hote} className={className} inert={inert} />;
});

/** Le lecteur de transcript, pour l'audio comme pour la vidéo.
 *
 *  `seek(seconds)` est exposé pour que la transcription puisse y mener ;
 *  `onTime` lui renvoie l'instant écouté. L'image n'apparaît que s'il y
 *  en a une.
 */
export const MediaPlayer = forwardRef(function MediaPlayer({ src, autoPlay = false, onTime }, ref) {
  return <EknAudio ref={ref} src={src} autoPlay={autoPlay} onTime={onTime} className="mt-3" />;
});
