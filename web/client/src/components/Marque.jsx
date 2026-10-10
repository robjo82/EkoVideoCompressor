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
  const menu = useRef(null);

  useEffect(() => {
    api.recoveryState()
      .then((etat) => { setRecuperation(Boolean(etat.enabled)); setInvitation(Boolean(etat.prompt)); })
      .catch(() => {});
  }, []);

  const ecarterInvitation = () => {
    setInvitation(false);
    api.recoveryDismiss().catch(() => {});
  };

  if (!moi?.email) return <span className="ml-auto" />;
  const initiales = moi.email.slice(0, 2).toUpperCase();
  const choisir = (action) => () => { menu.current?.hidePopover(); action(); };

  return (
    <div className="relative ml-auto">
      <button
        type="button"
        popoverTarget="menu-compte"
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
                    className="ekn-button ekn-button--subtle ekn-button--compact">
              Plus tard
            </button>
            <button
              type="button"
              onClick={() => { ecarterInvitation(); surVue('recuperation'); }}
              className="ekn-button ekn-button--compact"
            >
              Voir ce qu’on trouve
            </button>
          </div>
          <p className="mt-2 text-ekn-sm text-ekn-text-muted">Toujours accessible depuis ce menu.</p>
        </div>
      ) : null}
      {/* Un popover natif : le navigateur le ferme sur Échap ou sur un clic
          ailleurs, et le place sous l'avatar. Îlot clair dans l'en-tête foncé. */}
      <div
        ref={menu}
        id="menu-compte"
        popover="auto"
        data-ekn-theme="light"
        className="ekn-menu"
        onToggle={(e) => setOuvert(e.newState === 'open')}
      >
        <div className="ekn-menu__header">
          <span>{moi.email}</span>
        </div>
        <button type="button" onClick={choisir(() => surVue('compte'))}>Réglages</button>
        {recuperation ? (
          <button type="button" onClick={choisir(() => surVue('recuperation'))}>
            Récupérer mon historique
          </button>
        ) : null}
        <button type="button" onClick={choisir(surAide)}>
          Raccourcis clavier <kbd className="ekn-key">?</kbd>
        </button>
        <hr />
        <a href={DECONNEXION}>Se déconnecter</a>
      </div>
    </div>
  );
}
