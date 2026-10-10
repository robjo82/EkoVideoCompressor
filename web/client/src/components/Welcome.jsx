import { useEffect, useState } from 'react';
import { api } from '../api.js';

/** L'accueil des nouveaux venus, en tête de la bibliothèque.
 *
 *  Trois premiers pas, cochés d'après ce que la personne a vraiment fait
 *  — pas d'après une case qu'elle aurait cochée. Il s'efface seul une fois
 *  l'essentiel fait, ou d'un clic. Le Mac est facultatif : tout le monde
 *  n'utilisait pas l'app.
 */
export function Welcome({ onNavigate }) {
  const [state, setState] = useState(null);

  useEffect(() => {
    api.onboarding().then(setState).catch(() => {});
  }, []);

  if (!state?.show) return null;
  const { steps } = state;

  const dismiss = () => {
    setState({ ...state, show: false });
    api.onboardingDismiss().catch(() => {});
  };

  return (
    <div className="verre mt-6 rounded-xl p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="titre text-[1.0625rem] font-medium">Bienvenue sur transcript</h2>
          <p className="mt-1 text-ekn-sm text-fonce/60">
            Trois premiers pas, et tes réunions se transcrivent, se rattachent au bon
            dossier Odoo et s’y déposent toutes seules.
          </p>
        </div>
        <button
          type="button"
          onClick={dismiss}
          className="shrink-0 text-ekn-sm text-ekn-text-muted hover:text-fonce"
        >
          Masquer
        </button>
      </div>

      <ol className="mt-4 space-y-3">
        <Step
          done={steps.odoo}
          title="Enregistrer ta clé API Odoo"
          text="C’est elle qui permet de retrouver le dossier d’une réunion et d’y déposer la transcription, à ton nom."
          action={steps.odoo ? null : ['Ajouter ma clé', () => onNavigate('compte')]}
        />
        <Step
          done={steps.mac}
          optional
          title="Récupérer ta bibliothèque de l’app Mac"
          text={
            <>
              Si tu utilisais EkoVideo Compressor : accepte la mise à jour qu’il te propose
              (<em>Mettre à jour</em>), puis clique sur <em>Transférer ma bibliothèque</em> dans
              sa bannière. Ton navigateur s’ouvre : vérifie le code, puis <em>Autoriser</em>.
            </>
          }
        />
        <Step
          done={steps.transcription}
          title="Lancer ta première transcription"
          text="Glisse un enregistrement — ou plusieurs — sur « Nouvelle transcription ». Ton fichier reste sur ton poste."
          action={steps.transcription ? null : ['Commencer', () => onNavigate('nouveau')]}
        />
      </ol>
      <p className="mt-4 text-ekn-sm text-ekn-text-muted">
        Les raccourcis clavier et les réglages sont dans le menu de ton profil, en haut à droite.
      </p>
    </div>
  );
}

function Step({ done, optional = false, title, text, action = null }) {
  return (
    <li className="flex items-start gap-3">
      <span
        aria-hidden
        className={`mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full text-ekn-xs ${
          done ? 'bg-turquoise text-fonce' : 'ring-1 ring-bord'
        }`}
      >
        {done ? '✓' : ''}
      </span>
      <div className="min-w-0 flex-1">
        <p className={`text-[0.9375rem] ${done ? 'text-ekn-text-muted line-through decoration-fonce/25' : 'font-medium'}`}>
          {title}
          {optional && !done ? <span className="ml-2 text-ekn-xs font-normal text-ekn-text-muted">facultatif</span> : null}
        </p>
        {!done ? <p className="mt-0.5 text-ekn-sm leading-relaxed text-fonce/60">{text}</p> : null}
      </div>
      {action ? (
        <button
          type="button"
          onClick={action[1]}
          className="ekn-button ekn-button--compact shrink-0"
        >
          {action[0]}
        </button>
      ) : null}
    </li>
  );
}
