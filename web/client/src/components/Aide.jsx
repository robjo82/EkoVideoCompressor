import { useEffect } from 'react';
import { ALT, MOD } from '../raccourcis.js';

const RACCOURCIS = [
  ['Bibliothèque', [
    [[MOD, 'F'], 'Rechercher dans les transcriptions'],
    [['/'], 'Rechercher dans les transcriptions'],
    [['N'], 'Nouvelle transcription'],
    [[ALT, 'clic'], 'Sélectionner plusieurs réunions'],
    [['Maj', 'clic'], 'Sélectionner une plage'],
    [[MOD, 'A'], 'Tout sélectionner'],
    [['Échap'], 'Annuler la sélection'],
  ]],
  ['Transcription ouverte', [
    [[MOD, 'F'], 'Chercher dans la transcription'],
    [['Entrée'], 'Occurrence suivante'],
    [['Maj', 'Entrée'], 'Occurrence précédente'],
    [['Échap'], 'Fermer la recherche, puis revenir à la bibliothèque'],
  ]],
  ['Partout', [
    [['?'], 'Afficher ou masquer cette aide'],
  ]],
];

/** La liste des raccourcis, derrière « ? » : ils ne s'imposent à
 *  personne, mais qui les cherche les trouve. */
export function Aide({ surFermer }) {
  useEffect(() => {
    const echap = (e) => { if (e.key === 'Escape') surFermer(); };
    window.addEventListener('keydown', echap);
    return () => window.removeEventListener('keydown', echap);
  }, [surFermer]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-fonce/30 p-4 backdrop-blur-[2px]"
      onMouseDown={(e) => { if (e.target === e.currentTarget) surFermer(); }}
    >
      <div role="dialog" aria-modal="true" aria-label="Raccourcis clavier"
           className="w-full max-w-md rounded-2xl bg-white p-5 shadow-2xl">
        <div className="flex items-center justify-between">
          <p className="titre text-[1.0625rem] font-medium">Raccourcis clavier</p>
          <button type="button" onClick={surFermer} aria-label="Fermer"
                  className="rounded-md px-2 text-[1.25rem] leading-none text-ekn-text-muted hover:bg-papier hover:text-fonce">
            ×
          </button>
        </div>
        {RACCOURCIS.map(([groupe, lignes]) => (
          <div key={groupe} className="mt-4">
            <p className="text-ekn-xs font-medium uppercase tracking-wide text-ekn-text-muted">{groupe}</p>
            <ul className="mt-1.5 space-y-1.5">
              {lignes.map(([touches, libelle]) => (
                <li key={libelle + touches.join()} className="flex items-center justify-between gap-4 text-ekn-sm">
                  <span className="text-fonce/75">{libelle}</span>
                  <span className="flex shrink-0 gap-1">
                    {touches.map((t) => <Touche key={t}>{t}</Touche>)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Touche({ children }) {
  return (
    <kbd className="rounded border border-bord bg-papier px-1.5 py-0.5 font-sans text-ekn-xs text-fonce/70">
      {children}
    </kbd>
  );
}
