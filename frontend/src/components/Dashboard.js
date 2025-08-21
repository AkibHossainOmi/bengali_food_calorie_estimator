import React, { useState, useEffect, useContext } from "react";
import { foodAPI, calorieAPI } from "../utils/apiService";
import { AuthContext } from "../context/AuthContext";
import FoodSuggestions from "./FoodSuggestions";

// Helper to format food names
const formatFoodName = (rawName) => {
  const name = rawName.replaceAll("_", " ");
  return name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
};

export default function Dashboard() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [amount, setAmount] = useState("");
  const [dailyCalories, setDailyCalories] = useState(0);
  const [calorieGoal, setCalorieGoal] = useState(2000);
  const [recentFoods, setRecentFoods] = useState([]);
  const { user } = useContext(AuthContext);
  const [initialLoading, setInitialLoading] = useState(true);

  // Fetch daily progress on mount
  useEffect(() => {
    const fetchDailyProgress = async () => {
      const start = Date.now();
      try {
        const data = await calorieAPI.getDailyProgress();
        setDailyCalories(data.consumed || 0);
        setCalorieGoal(data.goal || 2000);
        setRecentFoods(
          data.recent_foods?.map((f) => ({
            name: formatFoodName(f.food_name),
            calories: f.calories,
            timestamp: f.predicted_at,
          })) || []
        );
      } catch (err) {
        console.error("Failed to fetch daily progress:", err);
      } finally {
        const elapsed = Date.now() - start;
        const remaining = Math.max(500 - elapsed, 0); // Ensure at least 2s
        setTimeout(() => setInitialLoading(false), remaining);
      }
    };
    fetchDailyProgress();
  }, []);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      if (selected.size > 5 * 1024 * 1024) {
        setError("File size exceeds 5MB limit. Please choose a smaller image.");
        return;
      }
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return setError("Please select an image first");
    if (!amount || amount <= 0) return setError("Please enter a valid amount in grams");

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await foodAPI.predictFood(file, amount);
      setResult(data); // show prediction but DO NOT add calories yet
    } catch (err) {
      setError(err.message || "Failed to analyze image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleAddToDaily = async () => {
    if (!result) return;
    try {
      // Send both calories and food name to backend
      await calorieAPI.addCalories(result.estimated_calories, result.food);

      setDailyCalories(prev => prev + result.estimated_calories);

      const newFood = {
        name: formatFoodName(result.food),
        calories: result.estimated_calories,
        timestamp: new Date().toISOString(),
      };
      setRecentFoods(prev => [newFood, ...prev.slice(0, 4)]);

      // Reset prediction and inputs
      setResult(null);
      setFile(null);
      setPreview(null);
      setAmount("");
      setError(null);
    } catch (err) {
      setError(err.message || "Failed to add calories. Please try again.");
    }
  };

  const progressPercentage = Math.min((dailyCalories / calorieGoal) * 100, 100);

  if (initialLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-r from-emerald-50 to-amber-50">
        <div className="w-12 h-12 border-4 border-gray-300 border-t-gray-500 rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-r from-emerald-50 to-amber-50">
      <div className="flex-1 flex flex-col min-w-0">
        <main className="flex-1 p-6 space-y-6 overflow-y-auto max-w-7xl mx-auto w-full">

          {/* Welcome Header */}
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-800">Welcome back, {user?.name || 'User'}!</h1>
            <p className="text-gray-600 mt-2">Track your nutrition with AI-powered calorie estimation</p>
          </div>

          {/* Daily Calorie Progress */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <h3 className="text-xl font-bold mb-4 text-gray-800">Daily Calorie Progress</h3>
            <div className="mb-2 flex justify-between">
              <span className="text-gray-600">Consumed: {dailyCalories} kcal</span>
              <span className="text-gray-600">Goal: {calorieGoal} kcal</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4">
              <div
                className="bg-emerald-600 h-4 rounded-full transition-all duration-500 ease-out"
                style={{ width: `${progressPercentage}%` }}
              />
            </div>
            <p className="mt-2 text-sm text-gray-500">
              {progressPercentage >= 100
                ? "You've reached your daily goal! 🎉"
                : `${Math.round(calorieGoal - dailyCalories)} kcal remaining`}
            </p>
          </div>

          {/* Food Prediction Card */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <h3 className="text-xl font-bold mb-6 text-gray-800">Estimate Food Calories</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column - Image Upload & Preview */}
              <div className="space-y-4">
                <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-6 hover:border-emerald-400 transition-colors bg-gray-50/50 min-h-[200px]">
                  <input 
                    type="file" 
                    accept="image/*" 
                    onChange={handleFileChange} 
                    className="hidden" 
                    id="file-upload" 
                  />
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer bg-white hover:bg-gray-100 text-gray-700 font-medium py-3 px-6 rounded-lg shadow-sm transition-all duration-200 border border-gray-300 flex items-center justify-center space-x-2"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M4 5a2 2 0 00-2 2v8a2 2 0 002 2h12a2 2 0 002-2V7a2 2 0 00-2-2h-1.586a1 1 0 01-.707-.293l-1.121-1.121A2 2 0 0011.172 3H8.828a2 2 0 00-1.414.586L6.293 4.707A1 1 0 015.586 5H4zm6 9a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                    </svg>
                    <span>{file ? "Change Image" : "Select Food Image"}</span>
                  </label>
                  <p className="text-gray-500 text-sm mt-3 text-center">Supported formats: JPG, PNG, or WEBP<br />Max file size: 5MB</p>
                </div>

                {preview && (
                  <div className="relative group">
                    <img 
                      src={preview} 
                      alt="Food preview" 
                      className="w-full h-64 object-cover rounded-lg shadow-sm border border-gray-200" 
                    />
                    <button 
                      type="button" 
                      onClick={() => { setPreview(null); setFile(null); setResult(null); }} 
                      className="absolute top-2 right-2 bg-white p-2 rounded-full shadow-md hover:bg-gray-100 transition-colors opacity-0 group-hover:opacity-100"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                )}
              </div>

              {/* Right Column - Form Inputs & Results */}
              <div className="space-y-6">
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div>
                    <label className="text-gray-700 font-medium mb-2 block">Amount (grams)</label>
                    <div className="relative">
                      <input 
                        type="number" 
                        value={amount} 
                        onChange={(e) => setAmount(e.target.value)} 
                        placeholder="Enter amount in grams" 
                        className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 focus:outline-none transition-colors" 
                        min="1" 
                      />
                    </div>
                  </div>

                  <button 
                    type="submit" 
                    disabled={loading || !file} 
                    className={`w-full py-3.5 rounded-lg font-semibold text-white transition-all duration-300 flex items-center justify-center space-x-2 ${
                      loading || !file ? 'bg-gray-400 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-700 shadow-md hover:shadow-lg'
                    }`}
                  >
                    {loading ? (
                      <>
                        <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M12.395 2.553a1 1 0 00-1.45-.385c-.345.23-.614.558-.822.88-.214.33-.403.713-.57 1.116-.334.804-.614 1.768-.84 2.734a31.365 31.365 0 00-.613 3.58 2.64 2.64 0 01-.945-1.067c-.328-.68-.398-1.534-.398-2.654A1 1 0 005.05 6.05 6.981 6.981 0 003 11a7 7 0 1011.95-4.95c-.592-.591-.98-.985-1.348-1.467-.363-.476-.724-1.063-1.207-2.03zM12.12 15.12A3 3 0 017 13s.879.5 2.5.5c0-1 .5-4 1.25-4.5.5 1 .786 1.293 1.371 1.879A2.99 2.99 0 0113 13a2.99 2.99 0 01-.879 2.121z" clipRule="evenodd" />
                        </svg>
                        <span>Estimate Calories</span>
                      </>
                    )}
                  </button>
                </form>

                {error && (
                  <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-start">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 mt-0.5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    <span>{error}</span>
                  </div>
                )}

                {result && (
                  <div className="mt-4 bg-emerald-50 p-5 rounded-lg border border-emerald-100">
                    <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-emerald-600" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      Analysis Complete
                    </h4>
                    
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm">
                        <p className="text-sm text-gray-500">Food Item</p>
                        <p className="font-medium">{formatFoodName(result.food)}</p>
                      </div>
                      <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm">
                        <p className="text-sm text-gray-500">Calories</p>
                        <p className="font-medium text-emerald-600">{result.estimated_calories} kcal</p>
                      </div>
                      <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm">
                        <p className="text-sm text-gray-500">Confidence</p>
                        <p className="font-medium">{result.confidence.toFixed(1)}%</p>
                      </div>
                      <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm">
                        <p className="text-sm text-gray-500">Amount</p>
                        <p className="font-medium">{amount}g</p>
                      </div>
                    </div>

                    <button 
                      onClick={handleAddToDaily} 
                      className="w-full bg-emerald-600 hover:bg-emerald-700 text-white py-2.5 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
                      </svg>
                      <span>Add to Daily Total</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Recent Foods */}
          <div className="bg-white rounded-xl p-6 shadow border border-gray-200">
            <h5 className="font-semibold text-gray-800 mb-4 text-lg">Recently Added Foods</h5>
            {recentFoods.length > 0 ? (
              recentFoods.map((food, idx) => (
                <div key={idx} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg mb-2">
                  <span>{food.name}</span>
                  <span>{food.calories} kcal</span>
                </div>
              ))
            ) : (
              <p className="text-gray-500 text-center py-4">No foods added today yet</p>
            )}
          </div>

          <FoodSuggestions
            weight={user?.weight}        // in kg
            height={user?.height}        // in cm
            dailyCalorieGoal={user?.daily_calorie_goal || 2000}  // fallback to 2000 kcal
          />

        </main>
      </div>
    </div>
  );
}
