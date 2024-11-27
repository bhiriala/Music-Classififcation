import React, { useState } from 'react';
import { Button, TextField, Typography } from '@mui/material';
import axios from 'axios';

function UploadFile({ serviceType }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    
    // Optional: Add basic file validation
    if (selectedFile && selectedFile.type !== 'audio/wav') {
      setResult('Please select a .wav file');
      setFile(null);
    } else {
      setFile(selectedFile);
      setResult(''); // Clear any previous error message
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    console.log(formData);

    try {
      const response = await axios.post('http://localhost:5001/predict', formData, {
        headers: {
          'Accept': 'application/json',
        },
      });

      // Log the response to check its structure
      console.log(response.data); // Debugging the response

      const genre = response.data.genre;

      if (genre) {
        setResult(`Genre classified by ${serviceType}: ${genre}`);
      } else {
        setResult('No genre predicted.');
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      const errorMessage = error.response
        ? `Error: ${error.response.data.error}`
        : 'Error classifying genre. Please try again.';
      setResult(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: 20 }}>
      <TextField
        type="file"
        accept=".wav"
        onChange={handleFileChange}
        fullWidth
        InputLabelProps={{ shrink: true }}
        inputProps={{
            style: {
            height: '41px', 
            display: 'flex',
            alignItems: 'center',
            },
        }}
        style={{ marginBottom: 10 }}
      />

      <Button
        variant="contained"
        color="primary"
        onClick={handleSubmit}
        disabled={!file || loading}
      >
        {loading ? 'Classifying...' : 'Classify Genre'}
      </Button>
      {result && (
        <Typography variant="body1" style={{ marginTop: 10 }}>
          {result}
        </Typography>
      )}
    </div>
  );
}

export default UploadFile;
