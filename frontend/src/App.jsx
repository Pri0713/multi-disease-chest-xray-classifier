import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const [predictions, setPredictions] = useState(null);

  const [gradcam, setGradcam] = useState(null);
  const [gradcamClass, setGradcamClass] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // =====================================================
  // FILE SELECTION
  // =====================================================

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (!selectedFile) {
      return;
    }

    setFile(selectedFile);

    // Create preview
    const imageURL = URL.createObjectURL(selectedFile);
    setPreview(imageURL);

    // Clear previous results
    setPredictions(null);
    setGradcam(null);
    setGradcamClass(null);
    setError("");
  };

  // =====================================================
  // ANALYZE X-RAY
  // =====================================================

  const analyzeXray = async () => {
    if (!file) {
      setError("Please select a chest X-ray first.");
      return;
    }

    setLoading(true);
    setError("");

    // Clear old results
    setPredictions(null);
    setGradcam(null);
    setGradcamClass(null);

    try {
      // =================================================
      // 1. SEND IMAGE TO /predict
      // =================================================

      const predictionData = new FormData();

      predictionData.append("file", file);

      console.log("Sending image to /predict...");

      const predictionResponse = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: predictionData,
        }
      );

      console.log(
        "Prediction HTTP status:",
        predictionResponse.status
      );

      if (!predictionResponse.ok) {
        throw new Error(
          `Prediction request failed: ${predictionResponse.status}`
        );
      }

      // Get JSON response
      const predictionResult =
        await predictionResponse.json();

      console.log(
        "Prediction response:",
        predictionResult
      );

      // =================================================
      // HANDLE DIFFERENT POSSIBLE RESPONSE FORMATS
      // =================================================

      let predictionArray = null;

      // Format 1:
      // [
      //   {
      //     disease: "Atelectasis",
      //     probability: 0.0085
      //   }
      // ]

      if (Array.isArray(predictionResult)) {
        predictionArray = predictionResult;
      }

      // Format 2:
      // {
      //   predictions: [...]
      // }

      else if (
        predictionResult &&
        Array.isArray(predictionResult.predictions)
      ) {
        predictionArray = predictionResult.predictions;
      }

      // Format 3:
      // {
      //   results: [...]
      // }

      else if (
        predictionResult &&
        Array.isArray(predictionResult.results)
      ) {
        predictionArray = predictionResult.results;
      }

      // Format 4:
      // {
      //   data: [...]
      // }

      else if (
        predictionResult &&
        Array.isArray(predictionResult.data)
      ) {
        predictionArray = predictionResult.data;
      }

      // =================================================
      // IF NOTHING MATCHES
      // =================================================

      if (!predictionArray) {
        console.log(
          "Unexpected backend response:",
          predictionResult
        );

        throw new Error(
          `Unexpected prediction response: ${JSON.stringify(
            predictionResult
          )}`
        );
      }

      // Save predictions
      setPredictions(predictionArray);

      console.log(
        "Predictions successfully loaded:",
        predictionArray
      );

      // =================================================
      // 2. SEND IMAGE TO /gradcam
      // =================================================

      const gradcamData = new FormData();

      gradcamData.append("file", file);

      console.log("Sending image to /gradcam...");

      const gradcamResponse = await fetch(
        `${API_URL}/gradcam`,
        {
          method: "POST",
          body: gradcamData,
        }
      );

      console.log(
        "Grad-CAM HTTP status:",
        gradcamResponse.status
      );

      if (!gradcamResponse.ok) {
        throw new Error(
          `Grad-CAM request failed: ${gradcamResponse.status}`
        );
      }

      // =================================================
      // GET GRAD-CAM IMAGE
      // =================================================

      const gradcamBlob =
        await gradcamResponse.blob();

      const gradcamURL =
        URL.createObjectURL(gradcamBlob);

      setGradcam(gradcamURL);

      // =================================================
      // GET GRAD-CAM CLASS FROM HEADER
      // =================================================

      const className =
        gradcamResponse.headers.get(
          "X-GradCAM-Class"
        );

      if (className) {
        setGradcamClass(className);
      }

      console.log(
        "Grad-CAM successfully loaded."
      );
    }

    // ===================================================
    // ERROR HANDLING
    // ===================================================

    catch (err) {
      console.error(
        "Analysis error:",
        err
      );

      setError(
        err.message ||
          "Unable to analyze the X-ray. Make sure the backend is running."
      );
    }

    finally {
      setLoading(false);
    }
  };

  // =====================================================
  // UI
  // =====================================================

  return (
    <div className="app">

      {/* =================================================
          HEADER
      ================================================= */}

      <header className="header">

        <div className="header-content">

          <div className="title-icon">
            🩻
          </div>

          <div>

            <h1>
              Multi-Disease Chest X-ray Classifier
            </h1>

            <p>
              AI-assisted chest X-ray analysis with
              disease probability predictions and
              Grad-CAM visualization.
            </p>

          </div>

        </div>

      </header>


      {/* =================================================
          MAIN CONTENT
      ================================================= */}

      <main className="container">


        {/* =================================================
            1. UPLOAD
        ================================================= */}

        <section className="card upload-card">

          <div className="section-title">

            <span className="section-number">
              1
            </span>

            <div>

              <h2>
                Upload Chest X-ray
              </h2>

              <p>
                Select a chest X-ray image to begin
                analysis.
              </p>

            </div>

          </div>


          {/* FILE INPUT */}

          <input
            className="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />


          {/* IMAGE PREVIEW */}

          {preview && (

            <div className="preview">

              <h3>
                Selected X-ray
              </h3>

              <img
                src={preview}
                alt="Selected chest X-ray"
              />

            </div>

          )}


          {/* ANALYZE BUTTON */}

          <button
            className="analyze-button"
            onClick={analyzeXray}
            disabled={!file || loading}
          >

            {loading
              ? "Analyzing X-ray..."
              : "Analyze X-ray"}

          </button>


          {/* ERROR */}

          {error && (

            <div className="error">
              {error}
            </div>

          )}

        </section>


        {/* =================================================
            2. PREDICTION RESULTS
        ================================================= */}

        {predictions && (

          <section className="card results-card">

            <div className="section-title">

              <span className="section-number">
                2
              </span>

              <div>

                <h2>
                  Prediction Results
                </h2>

                <p>
                  Probability scores returned by the
                  classification model.
                </p>

              </div>

            </div>


            <div className="table-container">

              <table className="prediction-table">

                <thead>

                  <tr>

                    <th>
                      #
                    </th>

                    <th>
                      Disease
                    </th>

                    <th>
                      Probability
                    </th>

                    <th>
                      Score
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {predictions.map(
                    (item, index) => {

                      // Make sure probability
                      // is treated as a number

                      const probability =
                        Number(
                          item.probability
                        ) || 0;

                      const percentage =
                        probability * 100;

                      return (

                        <tr
                          key={
                            item.disease ||
                            index
                          }
                        >

                          {/* RANK */}

                          <td className="rank">
                            {index + 1}
                          </td>


                          {/* DISEASE */}

                          <td className="disease-name">

                            {item.disease ||
                              item.label ||
                              "Unknown"}

                          </td>


                          {/* PERCENTAGE */}

                          <td className="percentage">

                            {percentage.toFixed(
                              2
                            )}
                            %

                          </td>


                          {/* PROBABILITY BAR */}

                          <td className="bar-cell">

                            <div
                              className="probability-bar"
                            >

                              <div
                                className="probability-fill"
                                style={{
                                  width: `${Math.min(
                                    percentage,
                                    100
                                  )}%`,
                                }}
                              />

                            </div>

                          </td>

                        </tr>

                      );
                    }
                  )}

                </tbody>

              </table>

            </div>

          </section>

        )}


        {/* =================================================
            3. GRAD-CAM
        ================================================= */}

        {gradcam && (

          <section className="card gradcam-card">

            <div className="section-title">

              <span className="section-number">
                3
              </span>

              <div>

                <h2>
                  Grad-CAM Explanation
                </h2>

                <p>
                  Visualization showing the region
                  associated with the selected model
                  prediction.
                </p>

              </div>

            </div>


            {/* GRAD-CAM CLASS */}

            {gradcamClass && (

              <div className="gradcam-class">

                <span>
                  Class
                </span>

                <strong>
                  {gradcamClass}
                </strong>

              </div>

            )}


            {/* GRAD-CAM IMAGE */}

            <img
              className="gradcam"
              src={gradcam}
              alt="Grad-CAM visualization"
            />

          </section>

        )}


        {/* =================================================
            DISCLAIMER
        ================================================= */}

        <div className="disclaimer">

          <strong>
            Note:
          </strong>{" "}

          This system is intended for research and
          educational purposes only and is not intended
          to provide a medical diagnosis.

        </div>


      </main>

    </div>
  );
}

export default App;