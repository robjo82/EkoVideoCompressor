import { useEffect, useRef, useState } from 'react';

/** Le peu de vocabulaire visuel partagé.
 *
 *  Volontairement maigre : la charte met en garde contre la carte
 *  réflexe. Ici une carte ne sert qu'à un objet autonome — un
 *  traitement, une fenêtre, une version. Le reste tient à
 *  l'alignement et à l'espacement.
 */

export function Bouton({ variante = 'primaire', className = '', ...props }) {
  const styles = {
    primaire: 'bg-fonce text-clair hover:bg-fonce-doux disabled:opacity-40',
    accent: 'bg-turquoise text-fonce hover:bg-turquoise-sombre disabled:opacity-40',
    discret: 'border border-bord/80 bg-white/70 text-fonce hover:border-fonce/40 hover:bg-white disabled:opacity-40',
  }[variante];
  return (
    <button
      {...props}
      className={`titre rounded-lg px-4 py-2 text-[0.9375rem] font-medium transition-colors disabled:cursor-not-allowed ${styles} ${className}`}
    />
  );
}

export function Champ({ label, aide, ...props }) {
  return (
    <label className="block">
      <span className="mb-1 block text-ekn-sm text-fonce/65">{label}</span>
      <input
        {...props}
        className="w-full rounded-lg border border-bord/80 bg-white/70 px-3 py-2 text-fonce transition-colors placeholder:text-ekn-text-muted focus:bg-white"
      />
      {aide ? <span className="mt-1 block text-ekn-sm text-ekn-text-muted">{aide}</span> : null}
    </label>
  );
}

/** États d'un traitement. Le violet est un accent, pas un second
 *  turquoise : il ne sert qu'à signaler qu'une version antérieure
 *  existe — un repère rare, donc précieux. */
const ETATS = {
  en_attente: ['En attente', 'bg-fonce/8 text-fonce/70'],
  en_cours: ['En cours', 'bg-turquoise/20 text-fonce'],
  a_finaliser: ['À finaliser', 'bg-turquoise/20 text-fonce'],
  finalisation: ['Fusion…', 'bg-turquoise/20 text-fonce'],
  termine: ['Terminé', 'bg-turquoise text-fonce'],
  erreur: ['Erreur', 'bg-[#b3261e]/12 text-[#8c1d18]'],
  recovered: ['Récupérée', 'bg-violet/10 text-violet'],
  cancelled: ['Interrompue', 'bg-fonce/8 text-fonce/60'],
};

export function Etat({ valeur }) {
  const [libelle, style] = ETATS[valeur] || [valeur, 'bg-fonce/8 text-fonce/70'];
  return (
    <span className={`inline-block rounded-md px-2 py-0.5 text-ekn-sm font-medium ${style}`}>
      {libelle}
    </span>
  );
}

export function Erreur({ children }) {
  if (!children) return null;
  return (
    <p className="rounded-lg border border-[#b3261e]/25 bg-[#b3261e]/6 px-3 py-2 text-ekn-sm text-[#8c1d18]">
      {children}
    </p>
  );
}

export function Vide({ titre, children }) {
  return (
    <div className="py-16 text-center">
      <p className="titre text-[1.0625rem] font-medium text-fonce">{titre}</p>
      <p className="mx-auto mt-2 max-w-md text-fonce/60">{children}</p>
    </div>
  );
}

const MOIS = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet',
  'août', 'septembre', 'octobre', 'novembre', 'décembre'];
const JOURS = ['L', 'M', 'M', 'J', 'V', 'S', 'D'];

/** « 2026-09-25T08:10 » ↔ Date locale, sans passer par l'UTC. */
function lireLocal(valeur) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(valeur || '');
  return m ? new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) : null;
}
function ecrireLocal(d) {
  const deux = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${deux(d.getMonth() + 1)}-${deux(d.getDate())}T${deux(d.getHours())}:${deux(d.getMinutes())}`;
}

/** Date et heure d'une réunion, sans le sélecteur natif du navigateur.
 *
 *  Le natif dépareille et varie d'un navigateur à l'autre. Ici : la date
 *  écrite en toutes lettres, qu'on ouvre sur un petit calendrier et deux
 *  champs d'heure. Même format de valeur qu'un champ datetime-local.
 */
export function DateHeure({ valeur, surChange, disabled = false }) {
  const [ouvert, setOuvert] = useState(false);
  const date = lireLocal(valeur) || new Date();
  const [mois, setMois] = useState(new Date(date.getFullYear(), date.getMonth(), 1));
  const boite = useRef(null);

  useEffect(() => {
    if (!ouvert) return undefined;
    const fermer = (e) => { if (!boite.current?.contains(e.target)) setOuvert(false); };
    const echap = (e) => { if (e.key === 'Escape') setOuvert(false); };
    document.addEventListener('pointerdown', fermer);
    document.addEventListener('keydown', echap);
    return () => {
      document.removeEventListener('pointerdown', fermer);
      document.removeEventListener('keydown', echap);
    };
  }, [ouvert]);

  const changer = (champs) => {
    const d = new Date(date);
    if ('jour' in champs) d.setFullYear(champs.jour.getFullYear(), champs.jour.getMonth(), champs.jour.getDate());
    if ('heure' in champs) d.setHours(champs.heure);
    if ('minute' in champs) d.setMinutes(champs.minute);
    surChange(ecrireLocal(d));
  };

  const premier = (mois.getDay() + 6) % 7; // lundi en tête
  const nbJours = new Date(mois.getFullYear(), mois.getMonth() + 1, 0).getDate();
  const cases = [...Array(premier).fill(null), ...Array.from({ length: nbJours }, (_, i) => i + 1)];
  const memeJour = (j) => j && date.getFullYear() === mois.getFullYear()
    && date.getMonth() === mois.getMonth() && date.getDate() === j;

  const champHeure = (valeurCourante, max, cle) => (
    <input
      type="number"
      min={0}
      max={max}
      value={String(valeurCourante).padStart(2, '0')}
      onChange={(e) => {
        const n = Math.min(Math.max(parseInt(e.target.value || '0', 10), 0), max);
        changer({ [cle]: n });
      }}
      className="w-14 rounded-md border border-bord bg-white px-2 py-1 text-center tabular-nums [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none"
    />
  );

  return (
    <div ref={boite} className="relative inline-block">
      <button
        type="button"
        disabled={disabled}
        onClick={() => {
          setMois(new Date(date.getFullYear(), date.getMonth(), 1));
          setOuvert((o) => !o);
        }}
        className="rounded-md border border-bord bg-white px-3 py-1.5 text-left text-ekn-sm hover:border-fonce/40 disabled:opacity-50"
      >
        {lireLocal(valeur)
          ? date.toLocaleString('fr-FR', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })
            + ' · ' + date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
          : 'Choisir une date'}
      </button>
      {ouvert ? (
        <div className="absolute left-0 z-30 mt-2 w-72 rounded-xl border border-bord bg-white p-3 shadow-lg">
          <div className="flex items-center justify-between">
            <button type="button" aria-label="Mois précédent"
                    onClick={() => setMois(new Date(mois.getFullYear(), mois.getMonth() - 1, 1))}
                    className="rounded px-2 py-1 text-fonce/60 hover:bg-papier">‹</button>
            <span className="titre text-ekn-sm font-medium">
              {MOIS[mois.getMonth()]} {mois.getFullYear()}
            </span>
            <button type="button" aria-label="Mois suivant"
                    onClick={() => setMois(new Date(mois.getFullYear(), mois.getMonth() + 1, 1))}
                    className="rounded px-2 py-1 text-fonce/60 hover:bg-papier">›</button>
          </div>
          <div className="mt-2 grid grid-cols-7 gap-1 text-center text-ekn-xs text-ekn-text-muted">
            {JOURS.map((j, i) => <span key={i}>{j}</span>)}
          </div>
          <div className="mt-1 grid grid-cols-7 gap-1">
            {cases.map((j, i) => (j ? (
              <button
                key={i}
                type="button"
                onClick={() => changer({ jour: new Date(mois.getFullYear(), mois.getMonth(), j) })}
                className={`rounded-md py-1 text-ekn-sm tabular-nums ${
                  memeJour(j) ? 'bg-fonce text-clair' : 'hover:bg-turquoise/15'
                }`}
              >
                {j}
              </button>
            ) : <span key={i} />))}
          </div>
          <div className="mt-3 flex items-center justify-between border-t border-bord pt-3 text-ekn-sm">
            <span className="text-fonce/60">Heure</span>
            <span className="flex items-center gap-1">
              {champHeure(date.getHours(), 23, 'heure')}
              <span className="text-ekn-text-muted">:</span>
              {champHeure(date.getMinutes(), 59, 'minute')}
            </span>
            <button type="button" onClick={() => setOuvert(false)}
                    className="rounded-md bg-fonce px-3 py-1 text-ekn-sm text-clair">OK</button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
