import { forwardRef, useImperativeHandle, useRef, useState } from 'react';
import { horodatage } from '../format.js';

const SPEEDS = [1, 1.25, 1.5, 2];

/** Le lecteur de transcript, pour l'audio comme pour la vidéo.
 *
 *  Le même vocabulaire que « Écouter et rogner » : un bouton rond, une
 *  frise qu'on attrape, le temps. Pas de lecteur natif : il change d'un
 *  navigateur à l'autre et dépareille. L'image n'apparaît que s'il y en
 *  a une — un enregistrement audio n'a pas à réserver un rectangle noir.
 *
 *  `seek(seconds)` est exposé pour que la transcription puisse y mener ;
 *  `onTime` lui renvoie l'instant écouté.
 */
export const MediaPlayer = forwardRef(function MediaPlayer({ src, autoPlay = false, onTime }, ref) {
  const media = useRef(null);
  const track = useRef(null);
  const [duration, setDuration] = useState(0);
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hasImage, setHasImage] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [readable, setReadable] = useState(true);
  const [dragging, setDragging] = useState(false);

  const seek = (seconds) => {
    if (!media.current) return;
    const t = Math.min(Math.max(seconds, 0), duration || seconds);
    media.current.currentTime = t;
    setPosition(t);
  };

  useImperativeHandle(ref, () => ({
    seek: (seconds, play = true) => {
      seek(seconds);
      if (play) media.current?.play().catch(() => {});
    },
  }), [duration]);

  const timeUnder = (clientX) => {
    const box = track.current.getBoundingClientRect();
    return Math.min(Math.max((clientX - box.left) / box.width, 0), 1) * duration;
  };

  const toggle = () => {
    const m = media.current;
    if (!m) return;
    if (m.paused) m.play().catch(() => setReadable(false));
    else m.pause();
  };

  const nextSpeed = () => {
    const next = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
    setSpeed(next);
    if (media.current) media.current.playbackRate = next;
  };

  const pct = duration ? `${(position / duration) * 100}%` : '0%';

  return (
    <div className="verre mt-3 rounded-xl p-3">
      <video
        ref={media}
        src={src}
        autoPlay={autoPlay}
        playsInline
        preload="metadata"
        onLoadedMetadata={(e) => {
          setDuration(e.currentTarget.duration || 0);
          setHasImage(e.currentTarget.videoWidth > 0);
        }}
        onTimeUpdate={(e) => {
          if (dragging) return;
          setPosition(e.currentTarget.currentTime);
          onTime?.(e.currentTarget.currentTime);
        }}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onError={() => setReadable(false)}
        className={hasImage ? 'mb-3 aspect-video w-full rounded-lg bg-fonce' : 'hidden'}
      />
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={toggle}
          disabled={!readable}
          aria-label={playing ? 'Pause' : 'Lecture'}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-fonce text-clair transition-colors hover:bg-fonce-doux disabled:opacity-30"
        >
          {playing ? (
            <span className="flex gap-1" aria-hidden>
              <span className="h-3.5 w-1 rounded-sm bg-current" />
              <span className="h-3.5 w-1 rounded-sm bg-current" />
            </span>
          ) : (
            <span aria-hidden className="ml-0.5 h-0 w-0 border-y-[7px] border-l-[11px] border-y-transparent border-l-current" />
          )}
        </button>

        <div
          ref={track}
          role="slider"
          aria-label="Position dans l’enregistrement"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(position)}
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft') seek(position - (e.shiftKey ? 30 : 5));
            if (e.key === 'ArrowRight') seek(position + (e.shiftKey ? 30 : 5));
            if (e.key === ' ') { e.preventDefault(); toggle(); }
          }}
          onPointerDown={(e) => {
            if (!duration) return;
            e.currentTarget.setPointerCapture(e.pointerId);
            setDragging(true);
            seek(timeUnder(e.clientX));
          }}
          onPointerMove={(e) => { if (dragging) seek(timeUnder(e.clientX)); }}
          onPointerUp={() => { setDragging(false); onTime?.(position); }}
          className="relative h-8 flex-1 cursor-pointer select-none focus:outline-none focus-visible:ring-2 focus-visible:ring-ekn-focus"
        >
          <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-fonce/10" />
          <div className="absolute left-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-turquoise" style={{ width: pct }} />
          <div
            className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-turquoise-sombre shadow"
            style={{ left: pct }}
          />
        </div>

        <span className="shrink-0 text-ekn-sm tabular-nums text-fonce/60">
          {horodatage(position)} / {horodatage(duration)}
        </span>
        <button
          type="button"
          onClick={nextSpeed}
          title="Vitesse de lecture"
          className="w-12 shrink-0 rounded-md py-1 text-ekn-sm tabular-nums text-fonce/60 ring-1 ring-bord hover:bg-white/60 hover:text-fonce"
        >
          {speed}×
        </button>
      </div>
      {!readable ? (
        <p className="mt-2 text-ekn-sm text-ekn-text-muted">Format non lisible par ce navigateur.</p>
      ) : null}
    </div>
  );
});
