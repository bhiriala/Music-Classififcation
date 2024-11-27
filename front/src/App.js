import React from 'react';
import './App.css';

function App() {
  return (
    <div className="app">
      <div className="upload-container">
        <h1>Upload Your Song</h1>
        <form>
          <div className="form-group">
            <label htmlFor="song">Choose a Song</label>
            <input type="file" id="song" name="song" />
          </div>
          <div className="form-group">
            <label htmlFor="model">Select Model</label>
            <select id="model" name="model">
              <option value="svm">SVM</option>
              <option value="rf">VGG</option>
            </select>
          </div>
          <button type="submit" className="upload-button">Upload Song</button>
        </form>
      </div>
    </div>
  );
}

export default App;