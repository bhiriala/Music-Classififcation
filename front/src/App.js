import React, { useState } from 'react';
import './App.css';
import UploadFile from './components/UploadFile';
import { Button, Card, CardContent, Typography } from '@mui/material';

function App() {
  const [userInput, setUserInput] = useState('');

  return (
    <div className="app">
      <div className="upload-container">
        <h1>Upload Your Song</h1>
        <form>
          <div className="form-group">
            <label htmlFor="model">Select Model</label>
           
            <select
              id="model"
              name="model"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
            >
              <option value="" disabled>Select a model</option>
              <option value="svm">SVM</option>
              <option value="vgg">VGG</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="song">Choose a Song</label>
            <UploadFile userInput={userInput} />
          </div>
        </form>
      </div>
    </div>
  );
}

export default App;
