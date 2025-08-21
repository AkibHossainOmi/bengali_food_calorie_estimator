import React, { useState, useEffect, useContext } from "react";
import { AuthContext } from "../context/AuthContext";
import { profileAPI } from "../utils/apiService";

export default function Profile() {
  const { fetchUserProfile } = useContext(AuthContext); // <-- get user & context updater

  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null); // <-- success/fail message
  const [error, setError] = useState(null);
  const [isEditing, setIsEditing] = useState(false);

  const [formData, setFormData] = useState({
    name: "",
    age: "",
    gender: "",
    weight: "",
    height: "",
    activity_level: "",
  });

  const activityFactors = {
    sedentary: 1.2,
    light: 1.375,
    moderate: 1.55,
    active: 1.725,
    extra: 1.9,
  };

  const activityDescriptions = {
    sedentary: "Sedentary (little or no exercise)",
    light: "Light (1–3 days/week)",
    moderate: "Moderate (3–5 days/week)",
    active: "Active (6–7 days/week)",
    extra: "Extra Active (physical job / 2x training)",
  };

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await profileAPI.getProfile();
        const data = res.data || res;
        setProfile(data);
        setFormData({
          name: data.name || "",
          age: data.age || "",
          gender: data.gender || "",
          weight: data.weight || "",
          height: data.height || "",
          activity_level: data.activity_level || "",
        });
      } catch (err) {
        console.error("Failed to fetch profile:", err);
        setError("Unable to load profile.");
      } finally {
        setLoading(false);
      }
    };

    fetchProfile();
  }, []);

  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const calculateBmrCalories = () => {
    const { age, gender, weight, height, activity_level } = formData;
    if (!age || !gender || !weight || !height || !activity_level) return "";

    const w = parseFloat(weight);
    const h = parseFloat(height);
    const a = parseInt(age);

    let bmr = 0;
    if (gender === "male") bmr = 10 * w + 6.25 * h - 5 * a + 5;
    else if (gender === "female") bmr = 10 * w + 6.25 * h - 5 * a - 161;

    const factor = activityFactors[activity_level] || 1.2;
    return Math.round(bmr * factor); // Current calorie consumption
  };

  const calculateBmiGoal = () => {
    const { weight, height } = formData;
    if (!weight || !height) return "";

    const w = parseFloat(weight);
    const h = parseFloat(height) / 100;
    const bmi = w / (h * h);

    let goal = calculateBmrCalories(); // start with current calories
    if (bmi < 18.5) goal = Math.round(goal * 1.1); // underweight → increase
    else if (bmi >= 25 && bmi < 30) goal = Math.round(goal * 0.9); // overweight → reduce
    else if (bmi >= 30) goal = Math.round(goal * 0.8); // obese → reduce more
    // normal → keep as is
    return goal;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setMessage(null);

    try {
      const calorieGoal = calculateBmiGoal();
      console.log(calorieGoal);
      const res = await profileAPI.updateProfile({
        ...formData,
        daily_calorie_goal: calorieGoal,
      });

      const updatedProfile = res.data || res;
      setProfile(updatedProfile);
      setIsEditing(false);
      fetchUserProfile();
      setMessage("Profile updated successfully!");
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to update profile.");
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    if (!profile) return;
    setFormData({
      name: profile.name || "",
      age: profile.age || "",
      gender: profile.gender || "",
      weight: profile.weight || "",
      height: profile.height || "",
      activity_level: profile.activity_level || "",
    });
    setIsEditing(false);
    setError(null);
    setMessage(null);
  };

  if (loading)
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-emerald-600"></div>
      </div>
    );

  return (
    <div className="max-h-screen flex flex-col bg-gradient-to-r from-emerald-50 to-amber-50">
      <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md mt-20 mb-20">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-800">My Profile</h1>
          {!isEditing && (
            <button
              onClick={() => setIsEditing(true)}
              className="bg-emerald-600 text-white px-4 py-2 rounded-lg hover:bg-emerald-700 flex items-center"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              Edit Profile
            </button>
          )}
        </div>

        {message && (
          <div className="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 mb-6" role="alert">
            <p>{message}</p>
          </div>
        )}

        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
            <p>{error}</p>
          </div>
        )}

       {!isEditing ? (
          // View Mode
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 p-6 rounded-lg">
              <h2 className="text-xl font-semibold text-gray-700 mb-4 pb-2 border-b">Personal Information</h2>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium text-gray-500">Name</p>
                  <p className="text-lg">{profile.name || "Not provided"}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Age</p>
                  <p className="text-lg">{profile.age || "Not provided"}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Gender</p>
                  <p className="text-lg capitalize">{profile.gender || "Not provided"}</p>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 p-6 rounded-lg">
              <h2 className="text-xl font-semibold text-gray-700 mb-4 pb-2 border-b">Health Metrics</h2>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium text-gray-500">Weight</p>
                  <p className="text-lg">{profile.weight ? `${profile.weight} kg` : "Not provided"}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Height</p>
                  <p className="text-lg">{profile.height ? `${profile.height} cm` : "Not provided"}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Activity Level</p>
                  <p className="text-lg">{profile.activity_level ? activityDescriptions[profile.activity_level] : "Not provided"}</p>
                </div>
              </div>
            </div>

            <div className="md:col-span-2 bg-white p-6 rounded-lg border border-gray-200 shadow-xs">
              <div className="flex items-center mb-5">
                <div className="w-10 h-10 rounded-lg bg-emerald-100 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
                  </svg>
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Nutrition Goals</h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Current Calorie Consumption */}
                <div className="bg-emerald-50 p-5 rounded-lg border border-emerald-100">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-medium text-emerald-800 uppercase tracking-wide">Current Calorie Consumption</h3>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                  </div>
                  
                  <div className="flex items-baseline mb-2">
                    <span className="text-2xl font-bold text-emerald-700 mr-2">
                      {calculateBmrCalories() || "0"}
                    </span>
                    <span className="text-sm text-emerald-600 font-medium">kcal</span>
                  </div>
                  
                  <div className="flex items-center text-xs text-emerald-600 mt-3">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>Based on your BMR and activity level</span>
                  </div>
                </div>

                {/* BMI-Adjusted Calorie Goal */}
                <div className="bg-blue-50 p-5 rounded-lg border border-blue-100">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-medium text-blue-800 uppercase tracking-wide">Recommended Calorie Goal</h3>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  
                  <div className="flex items-baseline mb-2">
                    <span className="text-2xl font-bold text-blue-700 mr-2">
                      {calculateBmiGoal() || "0"}
                    </span>
                    <span className="text-sm text-blue-600 font-medium">kcal</span>
                  </div>
                  
                  <div className="flex items-center text-xs text-blue-600 mt-3">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>Adjusted for your BMI to reach a healthy weight</span>
                  </div>
                </div>
              </div>

              {/* Additional Information */}
              <div className="mt-6 pt-5 border-t border-gray-100">
                <div className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-gray-400 mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-xs text-gray-500">
                    These calculations are estimates based on the Harris-Benedict equation. Individual needs may vary based on metabolism, genetics, and other factors.
                    {calculateBmiGoal() && calculateBmrCalories() && (
                      <span className="block mt-1 font-medium">
                        Recommended adjustment: {calculateBmiGoal() - calculateBmrCalories() > 0 ? "+" : ""}{calculateBmiGoal() - calculateBmrCalories()} kcal per day
                      </span>
                    )}
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          // Edit Mode
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-700">Personal Information</h3>
                
                {/* Name */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-emerald-500 focus:border-emerald-500"
                    placeholder="Your name"
                  />
                </div>

                {/* Age */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
                  <input
                    type="number"
                    name="age"
                    value={formData.age}
                    onChange={handleChange}
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-emerald-500 focus:border-emerald-500"
                    placeholder="Your age"
                    min="1"
                  />
                </div>

                {/* Gender */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
                  <select
                    name="gender"
                    value={formData.gender}
                    onChange={handleChange}
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-emerald-500 focus:border-emerald-500"
                  >
                    <option value="">Select Gender</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-700">Health Metrics</h3>
                
                {/* Weight */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Weight (kg)</label>
                  <input
                    type="number"
                    name="weight"
                    value={formData.weight}
                    onChange={handleChange}
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-emerald-500 focus:border-emerald-500"
                    placeholder="Your weight in kg"
                    min="1"
                    step="any"
                  />
                </div>

                {/* Height */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Height (cm)</label>
                  <input
                    type="number"
                    name="height"
                    value={formData.height}
                    onChange={handleChange}
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-emerald-500 focus:border-emerald-500"
                    placeholder="Your height in cm"
                    min="1"
                    step="any"
                  />
                </div>

                {/* Activity Level */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Activity Level</label>
                  <select
                    name="activity_level"
                    value={formData.activity_level}
                    onChange={handleChange}
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-emerald-500 focus:border-emerald-500"
                  >
                    <option value="">Select Activity Level</option>
                    <option value="sedentary">Sedentary (little or no exercise)</option>
                    <option value="light">Light (1–3 days/week)</option>
                    <option value="moderate">Moderate (3–5 days/week)</option>
                    <option value="active">Active (6–7 days/week)</option>
                    <option value="extra">Extra Active (physical job / 2x training)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Daily Calorie Goal */}
            <div className="md:col-span-2 bg-emerald-50 p-6 rounded-lg">
              <h2 className="text-xl font-semibold text-gray-700 mb-4 pb-2 border-b">Nutrition Goals</h2>
              <div className="space-y-2">
                <div>
                  <p className="text-sm font-medium text-gray-500">Current Calorie Consumption</p>
                  <p className="text-2xl font-bold text-emerald-600">
                    {calculateBmrCalories() || "Not calculated"} kcal
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    Based on your BMR and activity level
                  </p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">BMI-Adjusted Calorie Goal</p>
                  <p className="text-2xl font-bold text-emerald-600">
                    {calculateBmiGoal() || "Not calculated"} kcal
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    Adjusted for your BMI to reach a healthy weight
                  </p>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3 pt-4">
              <button
                type="button"
                onClick={handleCancel}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={saving}
                className="bg-emerald-600 text-white px-4 py-2 rounded-lg hover:bg-emerald-700 disabled:opacity-60 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500 flex items-center"
              >
                {saving ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Saving...
                  </>
                ) : "Update Profile"}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
