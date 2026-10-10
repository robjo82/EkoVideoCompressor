import { useEffect, useRef } from 'react';
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
 *  personne, mais qui les cherche les trouve. Un <dialog> natif : le
 *  navigateur garde le focus dedans, ferme sur Échap et rend le focus au
 *  bouton qui l'a ouvert.
 *
 *  m-auto : .ekn-dialog compte sur la marge auto du navigateur pour se
 *  centrer, que le preflight de Tailwind remet à zéro (signalé à
 *  ekonum-ui). En dessous de 40rem, sa règle de feuille du bas l'emporte. */
export function Aide({ surFermer }) {
  const boite = useRef(null);

  useEffect(() => {
    const d = boite.current;
    if (!d.open) d.showModal();
  }, []);

  // Un clic sur le voile, hors du cadre, ferme comme Échap.
  const horsCadre = (e) => {
    const r = boite.current.getBoundingClientRect();
    return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
  };

  return (
    <dialog
      ref={boite}
      className="ekn-dialog m-auto"
      aria-labelledby="aide-titre"
      onClose={surFermer}
      onMouseDown={(e) => { if (e.target === e.currentTarget && horsCadre(e)) surFermer(); }}
    >
      <form method="dialog">
        <h2 id="aide-titre">Raccourcis clavier</h2>
        {RACCOURCIS.map(([groupe, lignes]) => (
          <section key={groupe} className="mt-5">
            <h3 className="text-ekn-xs font-semibold uppercase tracking-wide text-ekn-text-muted">{groupe}</h3>
            <dl className="ekn-shortcuts mt-1">
              {lignes.map(([touches, libelle]) => (
                <div key={libelle + touches.join()}>
                  <dt>{libelle}</dt>
                  <dd>{touches.map((t) => <Touche key={t}>{t}</Touche>)}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
        <div className="ekn-dialog__actions">
          <button className="ekn-button ekn-button--subtle">Fermer</button>
        </div>
      </form>
    </dialog>
  );
}

export function Touche({ children }) {
  return <kbd className="ekn-key">{children}</kbd>;
}
