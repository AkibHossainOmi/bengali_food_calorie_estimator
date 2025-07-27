import React, { useState } from "react";
import { foodAPI } from "../utils/apiService";

export default function Predictor() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await foodAPI.predictFood(file);
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to predict. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-64px)] bg-gray-50 py-12 px-4">
      <div className="max-w-md mx-auto bg-white rounded-xl shadow-sm overflow-hidden border border-gray-100">
        <div className="p-6">
          <div className="text-center mb-6">
            <h1 className="text-2xl font-bold text-gray-800 mb-2">Bengali Food Calorie Estimator</h1>
            <p className="text-gray-600">Upload an image to analyze nutritional content</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-emerald-500 transition-colors">
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className="h-12 w-12 text-gray-400 mb-2" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <label 
                htmlFor="file-upload" 
                className="cursor-pointer bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-2 px-4 rounded-md transition-colors"
              >
                {file ? file.name : "Choose an image"}
              </label>
            </div>

            {preview && (
              <div className="relative group">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-64 object-cover rounded-lg shadow-sm border border-gray-200"
                />
                <button
                  type="button"
                  onClick={() => {
                    setPreview(null);
                    setFile(null);
                  }}
                  className="absolute top-2 right-2 bg-white/90 hover:bg-white text-gray-800 p-2 rounded-full shadow-md transition-all opacity-0 group-hover:opacity-100"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !file}
              className={`w-full py-3 px-4 rounded-md font-medium text-white transition-colors duration-200 ${
                loading || !file
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-emerald-600 hover:bg-emerald-700 shadow-md'
              }`}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </span>
              ) : (
                'Predict Calories'
              )}
            </button>
          </form>

          {error && (
            <div className="mt-4 bg-red-50 border-l-4 border-red-500 p-4 rounded">
              <p className="text-red-700 font-medium">{error}</p>
            </div>
          )}

          {result && (
            <div className="mt-6 bg-gray-50 rounded-lg p-6 border border-gray-200">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 text-center">Analysis Results</h2>
              <div className="space-y-3">
                <div className="flex justify-between border-b border-gray-100 pb-2">
                  <span className="text-gray-600">Food Item:</span>
                  <span className="font-medium text-gray-800">{result.food}</span>
                </div>
                <div className="flex justify-between border-b border-gray-100 pb-2">
                  <span className="text-gray-600">Estimated Calories:</span>
                  <span className="font-medium text-emerald-600">{result.estimated_calories} kcal</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Confidence:</span>
                  <span className="font-medium text-gray-800">{(result.confidence).toFixed(2)}%</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}