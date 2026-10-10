import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';
// Les composants optionnels d'ekonum-ui (<ekn-audio>…), à la version vendue.
import '../vendor/ekonum-ui/ekonum-ui.js';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
