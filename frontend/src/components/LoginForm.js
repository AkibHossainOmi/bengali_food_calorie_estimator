import React, { useState } from 'react';
import { authAPI } from '../utils/apiService';

export default function LoginForm({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const data = await authAPI.login({ username: email, password });
      localStorage.setItem('access_token', data.access_token);
      onLogin();
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-emerald-50 to-amber-50 px-4">
      <form
        onSubmit={handleSubmit}
        className="bg-white shadow-sm rounded-lg p-8 w-full max-w-md space-y-5 border border-gray-200"
      >
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-1">Welcome back</h2>
          <p className="text-gray-500 text-sm">Sign in to your account</p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-md p-3">
            {error}
          </div>
        )}

        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
            Email address
          </label>
          <input
            id="email"
            type="email"
            value={email}
            required
            onChange={(e) => setEmail(e.target.value)}
            className="w-full px-3 py-2 border border-gray-200 rounded-md focus:ring-1 focus:ring-gray-300 focus:border-gray-300 transition duration-150 placeholder-gray-400"
            placeholder="your@email.com"
          />
        </div>

        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
            Password
          </label>
          <input
            id="password"
            type="password"
            value={password}
            required
            onChange={(e) => setPassword(e.target.value)}
            className="w-full px-3 py-2 border border-gray-200 rounded-md focus:ring-1 focus:ring-gray-300 focus:border-gray-300 transition duration-150 placeholder-gray-400"
            placeholder="••••••••"
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <input
              id="remember-me"
              name="remember-me"
              type="checkbox"
              className="h-4 w-4 text-gray-700 focus:ring-gray-300 border-gray-300 rounded"
            />
            <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-700">
              Remember me
            </label>
          </div>
          <a href="/forgot-password" className="text-sm text-gray-600 hover:text-gray-800 transition-colors">
            Forgot password?
          </a>
        </div>

        <button
          type="submit"
          className="w-full py-2 px-4 bg-gray-900 hover:bg-gray-800 text-white font-medium rounded-md shadow-sm transition-colors duration-150"
        >
          Sign in
        </button>

        <div className="text-center text-sm text-gray-500">
          Don't have an account?{' '}
          <a href="/register" className="font-medium text-gray-700 hover:text-gray-900 transition-colors">
            Sign up
          </a>
        </div>
      </form>
    </div>
  );
}