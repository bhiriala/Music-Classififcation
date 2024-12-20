import React, { useState, useEffect } from 'react';
import { Button, TextField, Typography } from '@mui/material';
import axios from 'axios';

function UploadFile({ userInput }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState('');
  const [port, setPort] = useState(0);
  const [serviceType, setServiceType] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (userInput === 'svm') {
      setPort(5001);
      setServiceType('svm');
    } else {
      setPort(5003);
      setServiceType('vgg');
    }
  }, [userInput]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    
    if (selectedFile && selectedFile.type !== 'audio/wav') {
      setResult('Please select a .wav file');
      setFile(null);
    } else {
      setFile(selectedFile);
      setResult('');
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`http://localhost:${port}/predict`, formData, {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'multipart/form-data',
        },
      });

      const genre = response.data.genre;
      console.log(response.data);
      setResult(genre ? `Genre classified by ${serviceType}: ${genre}` : 'No genre predicted.');
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
    <div>
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
        style={{ width: '47%', marginRight: '33%', marginLeft: '27%', marginTop: 10, backgroundColor: 'rgb(111, 10, 212)',color:'white' }}
      >
        {loading ? 'Classifying...' : 'Classify Genre'}
      </Button>

      {result && (
        <Typography variant="body1" style={{ marginTop: 20, marginRight: '26%', marginLeft: '35.5%' }}>
          {result}
        </Typography>
      )}
    </div>
  );
}

export default UploadFile;
