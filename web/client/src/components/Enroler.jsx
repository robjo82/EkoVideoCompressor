import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { Bouton, Erreur, Vide } from './Communs.jsx';

/** Validation d'un appareil.
 *
 *  L'app macOS ne peut pas s'authentifier toute seule : elle affiche un
 *  code, on l'ouvre ici — où Access a déjà fait le tri — et on valide.
 *  L'appareil repart avec son propre jeton, qu'on peut révoquer seul
 *  depuis « Mon compte ».
 *
 *  La page dit **quel** appareil demande et **quand** : valider à
 *  l'aveugle un code qu'on n'a pas provoqué est le seul risque du
 *  procédé.
 */
export function Enroler({ code, surFini }) {
  const [demande, setDemande] = useState(null);
  const [erreur, setErreur] = useState('');
  const [etat, setEtat] = useState('lecture');

  useEffect(() => {
    api.enrolement(code)
      .then((v) => { setDemande(v); setEtat('prete'); })
      .catch((e) => { setErreur(e.message); setEtat('perdue'); });
  }, [code]);

  if (etat === 'perdue') {
    return (
      <section className="mx-auto max-w-xl px-6 py-16">
        <Vide titre="Code inconnu ou expiré">
          Relance l'enrôlement depuis l'application : les codes ne vivent que
          quelques minutes.
        </Vide>
        <Erreur>{erreur}</Erreur>
      </section>
    );
  }

  if (etat === 'validee') {
    return (
      <section className="mx-auto max-w-xl px-6 py-16 text-center">
        <h1 className="titre text-[1.5rem] font-semibold">Appareil autorisé</h1>
        <p className="mt-3 text-fonce/70">
          Retourne dans l'application : elle récupère son accès toute seule et
          commence à envoyer ton historique.
        </p>
        <Bouton className="mt-6" onClick={surFini}>Revenir à la bibliothèque</Bouton>
      </section>
    );
  }

  return (
    <section className="mx-auto max-w-xl px-6 py-16">
      <h1 className="titre text-[1.5rem] font-semibold">Autoriser cet appareil ?</h1>
      <div className="verre mt-6 rounded-xl p-5">
        <p className="text-ekn-sm text-ekn-text-muted">Appareil</p>
        <p className="titre text-[1.0625rem] font-medium">
          {demande?.appareil || 'appareil inconnu'}
        </p>
        <p className="mt-3 text-ekn-sm text-ekn-text-muted">Code</p>
        <p className="titre text-[1.0625rem] font-medium tracking-wider">{code}</p>
      </div>
      <p className="mt-4 text-ekn-sm text-fonce/70">
        Vérifie que ce code est bien celui affiché sur ton Mac. En autorisant,
        cet appareil obtient son propre accès à ton compte — révocable à tout
        moment depuis « Mon compte ».
      </p>
      <Erreur>{erreur}</Erreur>
      <Bouton
        className="mt-5"
        disabled={etat !== 'prete'}
        onClick={async () => {
          setEtat('envoi');
          try { await api.approuverEnrolement(code); setEtat('validee'); }
          catch (e) { setErreur(e.message); setEtat('prete'); }
        }}
      >
        {etat === 'envoi' ? 'Autorisation…' : 'Autoriser'}
      </Bouton>
    </section>
  );
}
