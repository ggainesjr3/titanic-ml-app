import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [formData, setFormData] = useState({
    pclass: 3,
    sex: 0,
    age: 22,
    title: 1
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: parseFloat(value) });
  };

  const getPrediction = async () => {
    setLoading(true);
    try {
      // Connects to the FastAPI backend running on port 8000
      const response = await axios.post('http://localhost:8000/predict', formData);
      setPrediction(response.data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
      alert("Is the Python API running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '500px', margin: '50px auto', padding: '20px', border: '1px solid #ddd', borderRadius: '8px', fontFamily: 'Arial' }}>
      <h2 style={{ textAlign: 'center' }}>🚢 Titanic Survival Predictor</h2>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        <label>Passenger Class (1, 2, or 3):
          <input type="number" name="pclass" value={formData.pclass} onChange={handleInputChange} style={{ width: '100%' }} />
        </label>

        <label>Sex (0 = Male, 1 = Female):
          <input type="number" name="sex" value={formData.sex} onChange={handleInputChange} style={{ width: '100%' }} />
        </label>

        <label>Age:
          <input type="number" name="age" value={formData.age} onChange={handleInputChange} style={{ width: '100%' }} />
        </label>

        <label>Title (1:Mr, 2:Miss, 3:Mrs, 4:Master, 5:Other):
          <input type="number" name="title" value={formData.title} onChange={handleInputChange} style={{ width: '100%' }} />
        </label>

        <button onClick={getPrediction} disabled={loading} style={{ padding: '10px', background: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
          {loading ? "Calculating..." : "Predict Survival"}
        </button>
      </div>

      {prediction && (
        <div style={{ marginTop: '20px', padding: '15px', background: prediction.survived ? '#d4edda' : '#f8d7da', borderRadius: '4px', textAlign: 'center' }}>
          <h3>{prediction.survived ? "✅ Survived!" : "❌ Did Not Survive"}</h3>
          <p>Model Confidence: {(prediction.probability * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}

export default App;