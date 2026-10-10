import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { DECONNEXION } from './Marque.jsx';
import { Bouton, Champ, Erreur } from './Communs.jsx';

/** Mon compte : identité, raccordement Odoo, jetons d'API.
 *
 *  Le raccordement Odoo est **personnel et non partageable**. La note
 *  déposée dans le chatter porte l'identité du propriétaire de la clé :
 *  avec une clé commune, tout serait signé du même compte et
 *  l'attribution — la raison d'être du chatter — disparaîtrait.
 */
export function Compte() {
  const [moi, setMoi] = useState(null);
  const [odoo, setOdoo] = useState(null);
  const [erreur, setErreur] = useState('');
  const [note, setNote] = useState('');

  const recharger = () =>
    Promise.all([api.me(), api.odooStatus()])
      .then(([m, o]) => { setMoi(m); setOdoo(o); })
      .catch((e) => setErreur(e.message));

  useEffect(() => { recharger(); }, []);

  return (
    <section className="mx-auto max-w-3xl px-6 py-10">
      <h1 className="titre text-[1.5rem] font-semibold">Mon compte</h1>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
        <p className="text-fonce/70">
          {moi?.email}
          {moi ? <span className="text-ekn-text-muted"> · connecté via {moi.via}</span> : null}
        </p>
        <a
          href={DECONNEXION}
          className="rounded-md px-3 py-1.5 text-ekn-sm text-fonce/70 ring-1 ring-bord hover:bg-white/60 hover:text-fonce"
        >
          Se déconnecter
        </a>
      </div>

      <Erreur>{erreur}</Erreur>
      {note ? <p className="mt-4 text-ekn-sm text-ekn-success-dark">{note}</p> : null}

      <div className="verre mt-8 rounded-xl p-5">
        <h2 className="titre text-[1.0625rem] font-medium">Odoo</h2>
        {odoo?.server ? (
          <p className="mt-1 text-ekn-sm text-fonce/60">{odoo.server}</p>
        ) : (
          <p className="mt-1 text-ekn-sm text-fonce/60">
            Aucun serveur Odoo configuré côté application.
          </p>
        )}

        <p className="mt-3 text-ekn-sm leading-relaxed text-fonce/70">
          Ta clé est <strong>personnelle</strong> : la transcription déposée dans
          le chatter porte ton nom. Une clé partagée signerait tout du même
          compte, et l'attribution — ce pour quoi le chatter existe — serait
          perdue.
        </p>
        <p className="mt-1 text-ekn-sm text-ekn-text-muted">
          Dans Odoo : ton avatar → <em>Mon profil</em> → <em>Sécurité du compte</em>{' '}
          → <em>Nouvelle clé API</em>. Elle ne s'affiche qu'une fois.
        </p>

        {odoo?.configured ? (
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <span className="text-[0.9375rem]">
              Raccordé en tant que <strong>{odoo.login}</strong>
            </span>
            <Bouton
              variante="discret"
              onClick={async () => {
                try {
                  await api.odooForget();
                  setNote('Clé retirée.');
                  recharger();
                } catch (e) { setErreur(e.message); }
              }}
            >
              Retirer
            </Bouton>
          </div>
        ) : (
          <FormulaireOdoo
            possible={odoo?.chiffrement_disponible !== false}
            surPose={async (login, cle) => {
              try {
                await api.odooSave(login, cle);
                setNote('Clé enregistrée, chiffrée.');
                recharger();
              } catch (e) { setErreur(e.message); }
            }}
          />
        )}
      </div>
    </section>
  );
}

function FormulaireOdoo({ possible, surPose }) {
  const [login, setLogin] = useState('');
  const [cle, setCle] = useState('');

  if (!possible) {
    return (
      <p className="mt-4 rounded-lg bg-[#b3261e]/6 px-3 py-2 text-ekn-sm text-[#8c1d18]">
        Le serveur n'a pas de clé de chiffrement configurée, et refuse donc
        d'enregistrer une clé API en clair. À régler côté déploiement.
      </p>
    );
  }

  return (
    <div className="mt-4 grid gap-3 sm:grid-cols-2">
      <Champ
        label="Ton identifiant Odoo"
        placeholder="robin@ekonum.fr"
        value={login}
        onChange={(e) => setLogin(e.target.value)}
      />
      <Champ
        label="Clé API"
        type="password"
        placeholder="•••••••••••••"
        value={cle}
        onChange={(e) => setCle(e.target.value)}
      />
      <div className="sm:col-span-2">
        <Bouton
          variante="accent"
          disabled={!login.trim() || cle.trim().length < 8}
          onClick={() => { surPose(login, cle); setCle(''); }}
        >
          Enregistrer
        </Bouton>
      </div>
    </div>
  );
}
