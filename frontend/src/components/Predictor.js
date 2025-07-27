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
    <div className="min-h-[calc(100vh-64px)] py-12 px-4 bg-gradient-to-r from-emerald-50 to-amber-50">
      <div className="max-w-3xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Bengali Food <span className="text-emerald-600">Calorie Estimator</span>
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Discover the nutritional content of your favorite Bengali dishes with AI-powered analysis
          </p>
        </div>

        {/* Prediction Card */}
        <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
          <div className="p-6 md:p-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Enhanced File Upload */}
              <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-8 hover:border-emerald-400 transition-colors bg-gray-50/50">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  className="h-14 w-14 text-gray-400 mb-4" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
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
                  className="cursor-pointer bg-white hover:bg-gray-100 text-gray-700 font-medium py-3 px-6 rounded-lg shadow-sm transition-all duration-200 border border-gray-300"
                >
                  {file ? file.name : "Select Food Image"}
                </label>
                <p className="text-gray-500 text-sm mt-3">JPG, PNG, or WEBP (Max 5MB)</p>
              </div>

              {/* Preview with Enhanced UI */}
              {preview && (
                <div className="relative group">
                  <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  <img
                    src={preview}
                    alt="Food preview"
                    className="w-full h-72 object-cover rounded-lg shadow-sm border border-gray-200"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      setPreview(null);
                      setFile(null);
                    }}
                    className="absolute top-3 right-3 bg-white/90 hover:bg-white text-gray-800 p-2 rounded-full shadow-md transition-all transform hover:scale-110"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </button>
                </div>
              )}

              {/* Submit Button with Enhanced Loading State */}
              <button
                type="submit"
                disabled={loading || !file}
                className={`w-full py-3.5 px-6 rounded-lg font-semibold text-white transition-all duration-300 ${
                  loading || !file
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-emerald-600 hover:bg-emerald-700 shadow-md hover:shadow-lg transform hover:-translate-y-0.5'
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing Dish...
                  </span>
                ) : (
                  'Estimate Calories'
                )}
              </button>
            </form>

            {/* Error Display */}
            {error && (
              <div className="mt-6 bg-red-50 border-l-4 border-red-500 p-4 rounded-lg">
                <div className="flex">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-red-500 mt-0.5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <p className="text-red-700 font-medium">{error}</p>
                </div>
              </div>
            )}

            {/* Results Display - Enhanced */}
            {result && (
              <div className="mt-8 bg-gradient-to-br from-emerald-50 to-amber-50 rounded-xl p-6 border border-emerald-100 shadow-inner">
                <h2 className="text-2xl font-semibold text-gray-800 mb-5 text-center">
                  Nutritional Analysis
                </h2>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center pb-3 border-b border-emerald-100">
                    <div className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-emerald-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                      </svg>
                      <span className="text-gray-700">Identified Dish</span>
                    </div>
                    <span className="font-semibold text-gray-900">{result.food}</span>
                  </div>
                  
                  <div className="flex justify-between items-center pb-3 border-b border-emerald-100">
                    <div className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-amber-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                      </svg>
                      <span className="text-gray-700">Calories</span>
                    </div>
                    <span className="font-bold text-emerald-600">{result.estimated_calories} kcal</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <div className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                      </svg>
                      <span className="text-gray-700">Confidence Level</span>
                    </div>
                    <div className="flex items-center">
                      <span className="font-medium text-gray-800 mr-2">{(result.confidence).toFixed(1)}%</span>
                      <div className="w-16 bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-emerald-600 h-2.5 rounded-full" 
                          style={{ width: `${result.confidence}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-emerald-100">
                  <p className="text-sm text-gray-500 text-center">
                    Results are estimates. For precise nutritional information, consult a dietitian.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Tips Section */}
        {!result && (
          <div className="mt-12 bg-white rounded-xl shadow-sm p-6 border border-gray-200">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-amber-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Tips for Best Results
            </h3>
            <ul className="space-y-3 text-gray-600">
              <li className="flex items-start">
                <svg className="h-5 w-5 text-emerald-500 mt-0.5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Use clear, well-lit photos of single dishes</span>
              </li>
              <li className="flex items-start">
                <svg className="h-5 w-5 text-emerald-500 mt-0.5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Capture the dish from directly above when possible</span>
              </li>
              <li className="flex items-start">
                <svg className="h-5 w-5 text-emerald-500 mt-0.5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Include common serving utensils for scale reference</span>
              </li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}