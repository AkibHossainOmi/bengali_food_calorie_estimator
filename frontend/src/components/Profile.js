import React, { useState, useEffect, useContext } from "react";
import { AuthContext } from "../context/AuthContext";
import { profileAPI } from "../utils/apiService";

export default function Profile() {
  useContext(AuthContext);

  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
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

  // Activity multipliers
  const activityFactors = {
    sedentary: 1.2,
    light: 1.375,
    moderate: 1.55,
    active: 1.725,
    extra: 1.9,
  };

  // Activity level descriptions
  const activityDescriptions = {
    sedentary: "Sedentary (little or no exercise)",
    light: "Light (1–3 days/week)",
    moderate: "Moderate (3–5 days/week)",
    active: "Active (6–7 days/week)",
    extra: "Extra Active (physical job / 2x training)",
  };

  // Fetch user profile
  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await profileAPI.getProfile();
        const data = res.data;
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

  // Handle input changes
  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  // BMR + Activity level calorie calculation
  const calculateCalories = () => {
    const { age, gender, weight, height, activity_level } = formData;
    if (!age || !gender || !weight || !height || !activity_level) return "";

    const w = parseFloat(weight);
    const h = parseFloat(height);
    const a = parseInt(age);

    let bmr = 0;
    if (gender === "male") {
      bmr = 10 * w + 6.25 * h - 5 * a + 5;
    } else if (gender === "female") {
      bmr = 10 * w + 6.25 * h - 5 * a - 161;
    }

    const factor = activityFactors[activity_level] || 1.2;
    return Math.round(bmr * factor);
  };

  // Save profile
  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError(null);

    try {
      const calorieGoal = calculateCalories();
      const res = await profileAPI.updateProfile({
        ...formData,
        daily_calorie_goal: calorieGoal,
      });

      setProfile(res);
      setIsEditing(false);
      alert("Profile updated successfully!");
    } catch (err) {
      setError(err.message || "Failed to update profile.");
    } finally {
      setSaving(false);
    }
  };

  // Cancel editing
  const handleCancel = () => {
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
  };

  if (loading) return (
    <div className="flex justify-center items-center h-64">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-emerald-600"></div>
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md mt-6">
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

          <div className="md:col-span-2 bg-emerald-50 p-6 rounded-lg">
            <h2 className="text-xl font-semibold text-gray-700 mb-4 pb-2 border-b">Nutrition Goals</h2>
            <div>
              <p className="text-sm font-medium text-gray-500">Daily Calorie Goal</p>
              <p className="text-2xl font-bold text-emerald-600">
                {profile.daily_calorie_goal || calculateCalories() || "Not calculated"}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Calculated based on your BMR and activity level
              </p>
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
          <div className="bg-emerald-50 p-4 rounded-lg">
            <label className="block text-sm font-medium text-gray-700 mb-1">Daily Calorie Goal</label>
            <div className="flex items-center">
              <input
                type="text"
                value={calculateCalories() || "Complete all fields to calculate"}
                readOnly
                className="flex-grow border border-gray-300 rounded-lg px-4 py-2 bg-white focus:ring-emerald-500 focus:border-emerald-500 font-medium text-emerald-600"
              />
            </div>
            <p className="text-sm text-gray-500 mt-1">
              This is automatically calculated based on your BMR and activity level.
            </p>
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
  );
}