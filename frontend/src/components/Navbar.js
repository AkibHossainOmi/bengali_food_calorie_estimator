import React, { useContext, useState, useEffect, useRef } from "react";
import { Link, useLocation } from "react-router-dom";
import { AuthContext } from "../context/AuthContext";

export default function Navbar() {
  const location = useLocation();
  const { loggedIn, handleLogout, user } = useContext(AuthContext); // assume user object has 'name'
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  const linkClasses = (path) =>
    `px-4 py-2 rounded-md font-medium transition-all duration-200 flex items-center ${
      location.pathname === path
        ? "bg-gray-100 text-gray-900 font-semibold"
        : "text-gray-600 hover:bg-gray-50 hover:text-gray-800"
    }`;

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const getInitial = (name) => {
    return name ? name.charAt(0).toUpperCase() : "?";
  };

  return (
    <nav className="bg-white shadow-sm text-gray-700 px-6 py-3 sticky top-0 z-50 border-b border-gray-200">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Left side - Navigation Links */}
        <div className="flex items-center space-x-4">
          <Link to="/" className={linkClasses("/")}>
            Home
          </Link>
          {loggedIn && (
            <Link to="/dashboard" className={linkClasses("/dashboard")}>
              Dashboard
            </Link>
          )}
        </div>

        {/* Right side - Auth Links / Dropdown */}
        <div className="flex items-center space-x-3 relative" ref={dropdownRef}>
          {loggedIn ? (
            <div className="relative">
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="flex items-center space-x-2 focus:outline-none"
              >
                {/* Circle with user initial */}
                <div className="w-8 h-8 bg-white text-xs text-slate-700 font-bold rounded-full flex items-center justify-center uppercase border border-gray-300">
                  {getInitial(user?.name)}
                </div>
                <svg
                  className="h-4 w-4 text-gray-700"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {dropdownOpen && (
                <div className="absolute right-0 mt-2 w-40 bg-white border border-gray-200 rounded-md shadow-lg py-1 z-50">
                  <Link
                    to="/profile"
                    className="block px-4 py-2 text-gray-700 hover:bg-gray-100"
                    onClick={() => setDropdownOpen(false)}
                  >
                    Profile
                  </Link>
                  <button
                    onClick={handleLogout}
                    className="w-full text-left px-4 py-2 text-red-700 hover:bg-gray-100"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          ) : (
            <>
              <Link to="/login" className={`${linkClasses("/login")} border border-gray-200 hover:border-gray-300`}>
                Login
              </Link>
              <Link
                to="/register"
                className="px-4 py-2 rounded-md font-medium text-white bg-gray-900 hover:bg-gray-800 transition-colors duration-200"
              >
                Register
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
