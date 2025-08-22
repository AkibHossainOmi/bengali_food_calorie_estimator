import React, { useState } from "react";
import { authAPI } from "../utils/apiService";

export default function RegisterForm() {
  const [formData, setFormData] = useState({
    name: "",
    age: "",
    gender: "",
    weight: "",
    height: "",
    activity_level: "",
    email: "",
    password: "",
    confirm: "",
  });

  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  const activityFactors = {
    sedentary: 1.2,
    light: 1.375,
    moderate: 1.55,
    active: 1.725,
    extra: 1.9,
  };

  // Calorie calculation (BMR * activity level)
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

  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage(null);
    setError(null);

    if (formData.password !== formData.confirm) {
      setError("Passwords do not match");
      return;
    }

    setSaving(true);
    try {
      await authAPI.register({
        ...formData,
        daily_calorie_goal: calculateCalories(),
      });
      setMessage("Registration successful. Check your email to verify.");
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-emerald-50 to-amber-50 px-4">
      <form
        onSubmit={handleSubmit}
        className="bg-white shadow-sm rounded-lg p-8 w-full max-w-2xl space-y-6 border border-gray-200"
      >
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-1">Create an account</h2>
          <p className="text-gray-500 text-sm">Complete your details to get started</p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-md p-3">
            {error}
          </div>
        )}

        {message && (
          <div className="bg-green-50 border border-green-200 text-green-700 text-sm rounded-md p-3">
            {message}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Personal Info */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-700">Personal Information</h3>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
              <input
                name="name"
                value={formData.name}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-4 py-2"
                placeholder="Your name"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
              <input
                type="number"
                name="age"
                value={formData.age}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-4 py-2"
                placeholder="Your age"
                min="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-4 py-2"
              >
                <option value="">Select Gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
          </div>

          {/* Health Info */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-700">Health Metrics</h3>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Weight (kg)</label>
              <input
                type="number"
                name="weight"
                value={formData.weight}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-4 py-2"
                placeholder="Your weight in kg"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Height (cm)</label>
              <input
                type="number"
                name="height"
                value={formData.height}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-4 py-2"
                placeholder="Your height in cm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Activity Level</label>
              <select
                name="activity_level"
                value={formData.activity_level}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-4 py-2"
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

        {/* Account Info */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-700">Account Information</h3>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              className="w-full border border-gray-300 rounded-lg px-4 py-2"
              placeholder="your@email.com"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              className="w-full border border-gray-300 rounded-lg px-4 py-2"
              placeholder="••••••••"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
            <input
              type="password"
              name="confirm"
              value={formData.confirm}
              onChange={handleChange}
              className="w-full border border-gray-300 rounded-lg px-4 py-2"
              placeholder="••••••••"
              required
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={saving}
          className="w-full py-2 px-4 bg-gray-900 hover:bg-gray-800 text-white font-medium rounded-md shadow-sm"
        >
          {saving ? "Registering..." : "Register"}
        </button>

        <div className="text-center text-sm text-gray-500">
          Already have an account?{" "}
          <a href="/login" className="font-medium text-gray-700 hover:text-gray-900">
            Sign in
          </a>
        </div>
      </form>
    </div>
  );
}
