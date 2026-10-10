import { useEffect, useState } from 'react';
import { EknAudio } from './MediaPlayer.jsx';

/** Réécoute et rognage avant de lancer.
 *
 *  Deux besoins que l'app macOS couvrait : vérifier qu'on a le bon
 *  enregistrement, et couper le bavardage du début ou la demi-heure
 *  oubliée à la fin. Transcrire ce qu'on va jeter coûte de l'argent et
 *  pollue le transcript.
 *
 *  Le lecteur est <ekn-audio trim> d'ekonum-ui : la partie gardée
 *  surlignée, deux poignées, une lecture qui part de la dernière touchée.
 *
 *  Le fichier se lit par une URL d'objet : rien ne part au serveur.
 */
export function Apercu({ fichier, duree, debut, fin, surDebut, surFin, actif }) {
  const [url, setUrl] = useState('');
  const [lisible, setLisible] = useState(true);

  useEffect(() => {
    if (!fichier) return undefined;
    setLisible(true);
    // Une réunion qu'on retraite se lit là où sa vidéo est stockée.
    if (fichier.remote) {
      setUrl(`${fichier.remote.url}?purpose=processing`);
      return undefined;
    }
    const objet = URL.createObjectURL(fichier);
    setUrl(objet);
    // Sans révocation, chaque changement de fichier laisse le précédent
    // épinglé en mémoire par le navigateur.
    return () => URL.revokeObjectURL(objet);
  }, [fichier]);

  if (!fichier || !duree || !url) return null;

  // Le lecteur compte en secondes entières, la sonde au centième : une fin
  // ramenée au bout ne doit pas rogner la dernière fraction de seconde.
  const rogner = ({ start, end }) => {
    surDebut(start);
    surFin(end >= Math.floor(duree) ? duree : end);
  };

  return (
    <div className="mt-4">
      <EknAudio
        src={url}
        trim
        label="Écouter et rogner"
        start={debut}
        end={fin}
        onTrim={rogner}
        onUnreadable={() => setLisible(false)}
        inert={actif}
        className={actif ? 'opacity-60' : ''}
      />
      <p className="mt-2 text-ekn-sm text-ekn-text-muted">
        {lisible
          ? 'Seule la partie surlignée sera transcrite — et payée. La lecture part de la dernière poignée touchée.'
          : 'Ce navigateur ne lit pas ce format : l’enregistrement sera transcrit en entier.'}
      </p>
    </div>
  );
}
