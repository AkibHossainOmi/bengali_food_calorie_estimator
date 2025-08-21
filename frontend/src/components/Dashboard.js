import React, { useState, useEffect, useContext } from "react";
import { foodAPI, calorieAPI } from "../utils/apiService";
import { AuthContext } from "../context/AuthContext";

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
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="w-12 h-12 border-4 border-gray-300 border-t-gray-500 rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
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
            <form onSubmit={handleSubmit} className="space-y-6">

              {/* File Upload */}
              <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-8 hover:border-emerald-400 transition-colors bg-gray-50/50">
                <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" id="file-upload" />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer bg-white hover:bg-gray-100 text-gray-700 font-medium py-3 px-6 rounded-lg shadow-sm transition-all duration-200 border border-gray-300 flex items-center"
                >
                  {file ? file.name : "Select Food Image"}
                </label>
                <p className="text-gray-500 text-sm mt-3">JPG, PNG, or WEBP (Max 5MB)</p>
              </div>

              {preview && (
                <div className="relative">
                  <img src={preview} alt="Food preview" className="w-full h-72 object-cover rounded-lg shadow-sm border border-gray-200 mt-4" />
                  <button type="button" onClick={() => { setPreview(null); setFile(null); setResult(null); }} className="absolute top-2 right-2 bg-white p-2 rounded-full shadow hover:bg-gray-100">✕</button>
                </div>
              )}

              <div>
                <label className="text-gray-700 font-medium mb-2 block">Amount (grams)</label>
                <input type="number" value={amount} onChange={(e) => setAmount(e.target.value)} placeholder="Enter amount" className="w-full border border-gray-300 rounded-lg p-3 focus:ring-emerald-400 focus:outline-none" min="1" />
              </div>

              <button type="submit" disabled={loading || !file} className={`w-full py-3.5 rounded-lg font-semibold text-white transition-all duration-300 flex items-center justify-center ${loading || !file ? 'bg-gray-400 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-700 shadow-md hover:shadow-lg'}`}>
                {loading ? "Analyzing..." : "Estimate Calories"}
              </button>
            </form>

            {error && <p className="mt-4 text-red-600">{error}</p>}

            {result && (
              <div className="mt-6 bg-emerald-50 p-6 rounded-lg border border-emerald-100 shadow-inner">
                <h4 className="text-xl font-semibold text-gray-800 mb-4">Analysis Results</h4>
                <p><strong>Dish:</strong> {formatFoodName(result.food)}</p>
                <p><strong>Calories:</strong> {result.estimated_calories} kcal</p>
                <p><strong>Confidence:</strong> {result.confidence.toFixed(1)}%</p>
                <p><strong>Amount:</strong> {amount}g</p>

                <button onClick={handleAddToDaily} className="mt-4 bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg text-sm">
                  Add to Daily Total
                </button>
              </div>
            )}
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

        </main>
      </div>
    </div>
  );
}
