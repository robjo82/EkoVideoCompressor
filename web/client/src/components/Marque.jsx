import { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';

/** En-tête de l'application.
 *
 *  Navigation flottante en verre : c'est l'un des usages que la charte
 *  cite explicitement, et le seul endroit où une surface vitrée est
 *  posée sur toute la largeur. Elle suit le défilement, donc le fond
 *  bouge derrière elle — c'est ce mouvement qui rend la matière lisible
 *  plutôt que décorative.
 */
export function Entete({ vue, surVue, surAide }) {
  const [moi, setMoi] = useState(null);

  useEffect(() => {
    api.me().then(setMoi).catch(() => {});
  }, []);

  const onglets = [
    ['bibliotheque', 'Bibliothèque'],
    ['nouveau', 'Nouvelle transcription', 'Nouvelle'],
  ];

  return (
    <header className="verre-sombre ekn-dark sticky top-0 z-20 text-clair">
      <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3 sm:gap-6 sm:px-6">
        <img src="/marque/logo-sombre.svg" alt="Ekonum" className="h-7 w-auto shrink-0" />

        <nav className="flex gap-1">
          {onglets.map(([cle, libelle, court]) => (
            <button
              key={cle}
              onClick={() => surVue(cle)}
              aria-current={vue === cle ? 'page' : undefined}
              aria-label={court ? libelle : undefined}
              className={`titre whitespace-nowrap rounded-lg px-2.5 py-1.5 text-[0.9375rem] font-medium sm:px-3 transition-colors ${
                vue === cle
                  ? 'bg-turquoise text-fonce'
                  : 'text-clair/70 hover:bg-clair/10 hover:text-clair'
              }`}
            >
              {court ? (
                <>
                  <span className="sm:hidden">{court}</span>
                  <span className="hidden sm:inline">{libelle}</span>
                </>
              ) : libelle}
            </button>
          ))}
        </nav>

        <Compte moi={moi} surVue={surVue} surAide={surAide} actif={vue === 'compte'} />
      </div>
    </header>
  );
}

/** Le compte connecté.
 *
 *  Savoir sous quelle identité on travaille est la première chose qu'on
 *  cherche sur un outil d'équipe — surtout un outil qui dépense de
 *  l'argent et où le vocabulaire est partagé.
 */
/** La session est celle de Cloudflare Access, pas de l'application : se
 *  déconnecter, c'est la clore là-bas. La visite suivante repasse par la
 *  page de connexion Google — c'est le « se connecter ». */
export const DECONNEXION = '/cdn-cgi/access/logout';

/** Le compte, et ce qui ne sert pas tous les jours : réglages,
 *  raccourcis, déconnexion — rangés derrière l'avatar plutôt qu'affichés
 *  en permanence. */
function Compte({ moi, surVue, surAide, actif }) {
  const [ouvert, setOuvert] = useState(false);
  // L'invitation à récupérer son historique : une fois, près du compte,
  // jusqu'à ce qu'on la suive ou qu'on l'écarte. Ensuite, le menu.
  const [invitation, setInvitation] = useState(false);
  const [recuperation, setRecuperation] = useState(false);
  const cadre = useRef(null);

  useEffect(() => {
    api.recoveryState()
      .then((etat) => { setRecuperation(Boolean(etat.enabled)); setInvitation(Boolean(etat.prompt)); })
      .catch(() => {});
  }, []);

  const ecarterInvitation = () => {
    setInvitation(false);
    api.recoveryDismiss().catch(() => {});
  };

  useEffect(() => {
    if (!ouvert) return undefined;
    const dehors = (e) => { if (!cadre.current?.contains(e.target)) setOuvert(false); };
    const echap = (e) => { if (e.key === 'Escape') setOuvert(false); };
    document.addEventListener('mousedown', dehors);
    window.addEventListener('keydown', echap);
    return () => {
      document.removeEventListener('mousedown', dehors);
      window.removeEventListener('keydown', echap);
    };
  }, [ouvert]);

  if (!moi?.email) return <span className="ml-auto" />;
  const initiales = moi.email.slice(0, 2).toUpperCase();
  const choisir = (action) => () => { setOuvert(false); action(); };
  const entree = 'block w-full rounded-md px-3 py-2 text-left text-ekn-sm text-fonce/80 transition-colors hover:bg-papier hover:text-fonce';

  return (
    <div ref={cadre} className="relative ml-auto">
      <button
        type="button"
        onClick={() => setOuvert((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={ouvert}
        title={`Connecté via ${moi.via}`}
        className={`flex items-center gap-2.5 rounded-full p-1 transition-colors sm:pl-3 ${
          actif || ouvert ? 'bg-clair/15' : 'hover:bg-clair/10'
        }`}
      >
        <span className="hidden text-ekn-sm text-clair/75 sm:inline">{moi.email}</span>
        <span
          aria-hidden
          className="titre grid h-8 w-8 place-items-center rounded-full bg-turquoise text-ekn-sm font-semibold text-fonce"
        >
          {initiales}
        </span>
      </button>
      {invitation && !ouvert ? (
        <div data-ekn-theme="light" className="absolute right-0 top-full z-30 mt-3 w-80 rounded-xl bg-white p-4 text-fonce shadow-2xl ring-1 ring-bord">
          <span aria-hidden className="absolute -top-1.5 right-4 h-3 w-3 rotate-45 bg-white ring-1 ring-bord [clip-path:polygon(0_0,100%_0,0_100%)]" />
          <p className="titre text-[0.9375rem] font-medium">Retrouver tes anciens enregistrements</p>
          <p className="mt-1 text-ekn-sm leading-relaxed text-fonce/65">
            Des réunions enregistrées avant transcript dorment peut-être dans ton
            Drive. On peut les retrouver et les ranger ici, à leur date — et faire
            le ménage derrière.
          </p>
          <div className="mt-3 flex items-center justify-end gap-2">
            <button type="button" onClick={ecarterInvitation}
                    className="rounded-md px-3 py-1.5 text-ekn-sm text-ekn-text-muted hover:text-fonce">
              Plus tard
            </button>
            <button
              type="button"
              onClick={() => { ecarterInvitation(); surVue('recuperation'); }}
              className="rounded-md bg-fonce px-3 py-1.5 text-ekn-sm text-clair hover:bg-fonce-doux"
            >
              Voir ce qu’on trouve
            </button>
          </div>
          <p className="mt-2 text-ekn-sm text-ekn-text-muted">Toujours accessible depuis ce menu.</p>
        </div>
      ) : null}
      {ouvert ? (
        <div
          role="menu"
          data-ekn-theme="light"
          className="absolute right-0 top-full z-30 mt-2 w-60 rounded-xl bg-white p-1.5 text-fonce shadow-2xl ring-1 ring-bord"
        >
          <p className="truncate px-3 pb-1.5 pt-1 text-ekn-xs text-ekn-text-muted">{moi.email}</p>
          <button type="button" role="menuitem" onClick={choisir(() => surVue('compte'))} className={entree}>
            Réglages
          </button>
          {recuperation ? (
            <button type="button" role="menuitem" onClick={choisir(() => surVue('recuperation'))} className={entree}>
              Récupérer mon historique
            </button>
          ) : null}
          <button type="button" role="menuitem" onClick={choisir(surAide)}
                  className={`${entree} flex items-center justify-between`}>
            Raccourcis clavier
            <kbd className="rounded border border-bord bg-papier px-1.5 text-ekn-xs text-ekn-text-muted">?</kbd>
          </button>
          <div className="my-1 h-px bg-bord/70" />
          <a role="menuitem" href={DECONNEXION} className={entree}>Se déconnecter</a>
        </div>
      ) : null}
    </div>
  );
}
