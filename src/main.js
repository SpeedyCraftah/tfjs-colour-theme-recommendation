import { useState } from 'react';
import './styles/main.css';
import { sequential, layers, train, tensor } from "@tensorflow/tfjs";

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1) + min);
}

function randomRGB() {
  return {
    r: randomInt(0, 255),
    g: randomInt(0, 255),
    b: randomInt(0, 255)
  };
}

function randomTheme() {
  return {
    textColour: randomRGB(),
    backgroundColour: randomRGB()
  };
}

const model = sequential();
model.add(layers.dense({
  inputShape: [6],
  activation: 'sigmoid',
  units: 24
}));
model.add(layers.dense({
  inputShape: [24],
  activation: 'sigmoid',
  units: 24
}));
model.add(layers.dense({
  inputShape: [24],
  activation: 'sigmoid',
  units: 1
}));
model.compile({
  loss: 'meanSquaredError',
  optimizer: train.adam()
});

let previousTrainingX = [];
let previousTrainingY = [];

function Main() {
  const [busy, setBusy] = useState(false);
  const [theme, setTheme] = useState({ textColour: randomRGB(), backgroundColour: randomRGB() });
  const [predictedThemes, setPredictedThemes] = useState([]);

  async function onRating(y) {
    setBusy(true);

    // Previous version of model (retraining whole set).
    /*const result = await model.fit(
      tensor([[
        theme.textColour.r / 255, theme.textColour.g / 255, theme.textColour.b / 255,
        theme.backgroundColour.r / 255, theme.backgroundColour.g / 255, theme.backgroundColour.b / 255
      ]], [1, 6]), tensor([y]), { epochs: 20 }
    );*/

    previousTrainingX.push([
      theme.textColour.r / 255, theme.textColour.g / 255, theme.textColour.b / 255,
      theme.backgroundColour.r / 255, theme.backgroundColour.g / 255, theme.backgroundColour.b / 255
    ]);
    previousTrainingY.push(y);

    if (previousTrainingX.length > 10) {
      previousTrainingX.shift();
      previousTrainingY.shift();
    }

    const result = await model.fit(
      tensor(previousTrainingX, [previousTrainingX.length, 6]), tensor(previousTrainingY), { epochs: 50, shuffle: true }
    );

    previousTrainingX = [];
    previousTrainingY = [];

    console.log(result);

    // Predict new themes.
    let newThemes = [];
    for (let i = 0; i < 12000; i++) {
      newThemes.push({
        score: 0,
        ...randomTheme()
      });
    }

    const scorePredictions = await model.predict(
      tensor(newThemes.map(t => {
        return [
          t.textColour.r / 255, t.textColour.g / 255, t.textColour.b / 255,
          t.backgroundColour.r / 255, t.backgroundColour.g / 255, t.backgroundColour.b / 255
        ]
      }), [12000, 6])
    ).data();

    for (let i = 0; i < scorePredictions.length; i++) {
      newThemes[i].score = scorePredictions[i];
    }

    // Sort the themes from highest score to lowest.
    newThemes.sort((a, b) => b.score - a.score);
    
    // Set new 50 recommended themes,.
    const bestThemes = newThemes.slice(0, 50);
    setPredictedThemes(bestThemes);

    // Predict either completely random theme or one from the top 50 best.
    setTheme(Math.random() < 0.5 ? randomTheme() : bestThemes[Math.floor(Math.random() * bestThemes.length)]);
    setBusy(false);
  }

  return (
    <div style={{height: "100%"}}>
      <title>ML Colour Picker</title>
      <h1 class="page-title">ML Colour Recommendation Engine</h1>

      <div style={{marginTop: "3em"}} class="rate-target-container">
        <div style={{backgroundColor: `rgb(${theme.backgroundColour.r}, ${theme.backgroundColour.g}, ${theme.backgroundColour.b})`}} class="rate-target">
          <p style={{color: `rgb(${theme.textColour.r}, ${theme.textColour.g}, ${theme.textColour.b})`, fontSize: '1.85em', fontFamily: 'Roboto sans-serif'}}>Hello, World!</p>
        </div>
        
        <div class="rate-container" style={{marginTop: "0.8em"}}>
          <button onClick={() => onRating(1)} disabled={busy} style={{backgroundColor: '#41b020'}}> This is clean</button>
          <button onClick={() => onRating(0.5)} disabled={busy} style={{backgroundColor: '#eda528'}}>Meh...</button>
          <button onClick={() => onRating(0)} disabled={busy} style={{backgroundColor: '#ed2424'}}>Absolute trash</button>
        </div>
      </div>

      <h2 style={{marginTop: '5em', textAlign: 'center' }}>Here are some themes you may like based on your ratings</h2>
      <div class="recommended-container" style={{marginTop: '1.7em'}}>
        {predictedThemes.length === 0 ? <p>There are no predicted themes yet. Try rating some themes!</p> : predictedThemes.map((d, i) => {
          return (
            <div title={`Assigned score: ${d.score}`} key={i} class="recommended-child" style={{backgroundColor: `rgb(${d.backgroundColour.r}, ${d.backgroundColour.g}, ${d.backgroundColour.b})`}}>
              <p style={{color: `rgb(${d.textColour.r}, ${d.textColour.g}, ${d.textColour.b})`, fontSize: '1.1em', fontFamily: 'Roboto sans-serif'}}>Hello, World!</p>
            </div>
          );
        })}
      </div>

    </div>
  );
}

export default Main;
