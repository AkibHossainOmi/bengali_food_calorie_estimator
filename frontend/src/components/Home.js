import { useState } from 'react';

export default function Home() {
  const [hoveredDish, setHoveredDish] = useState(null);
  
  const popularDishes = [
    { name: 'Shorshe Ilish', calories: '320' },
    { name: 'Chingri Malai Curry', calories: '280' },
    { name: 'Mutton Kosha', calories: '450' },
    { name: 'Mishti Doi', calories: '180' },
  ];

  return (
    <div className="min-h-[calc(100vh-64px)] bg-gradient-to-b from-gray-50 to-white">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-28">
          <div className="text-center relative z-10">
            <h1 className="text-4xl md:text-4xl font-bold text-gray-900 mb-6">
              Taste <span className="text-emerald-600">Authentic Bengal</span>,<br />Track <span className="text-amber-600">Every Calorie</span>
            </h1>
            <p className="text-xl text-gray-600 mb-10 max-w-3xl mx-auto">
              Our AI-powered analyzer helps you enjoy traditional Bengali cuisine while maintaining nutritional awareness.
            </p>
            <div className="flex flex-col sm:flex-row justify-center gap-4">
              <a 
                href="/predict" 
                className="px-8 py-4 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-lg shadow-md transition-all duration-300 hover:shadow-lg transform hover:-translate-y-1 flex items-center justify-center"
              >
                Start Predicting Now
                <svg xmlns="http://www.w3.org/2000/svg" className="ml-2 h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </a>
              <a 
                href="/about" 
                className="px-8 py-4 border-2 border-gray-300 text-gray-700 font-semibold rounded-lg hover:border-emerald-500 hover:text-emerald-600 transition-colors duration-300"
              >
                Learn How It Works
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Why Choose Our Estimator</h2>
            <div className="w-24 h-1 bg-emerald-500 mx-auto"></div>
          </div>
          
          <div className="grid md:grid-cols-3 gap-10">
            {[
              {
                title: "AI-Powered Accuracy",
                description: "Our advanced machine learning model provides precise calorie estimates for authentic Bengali recipes.",
                icon: (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                )
              },
              {
                title: "Comprehensive Database",
                description: "Access nutritional information for hundreds of traditional dishes from across Bengal.",
                icon: (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                  </svg>
                )
              },
              {
                title: "Health Insights",
                description: "Get personalized recommendations based on your dietary preferences and health goals.",
                icon: (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                  </svg>
                )
              }
            ].map((feature, index) => (
              <div 
                key={index} 
                className="bg-gray-50 p-8 rounded-xl hover:shadow-lg transition-shadow duration-300 border border-gray-100 text-center"
              >
                <div className="flex justify-center mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Popular Dishes Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Explore Popular Dishes</h2>
            <p className="text-gray-600 max-w-2xl mx-auto">Discover the nutritional content of these beloved Bengali classics</p>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {popularDishes.map((dish, index) => (
              <div 
                key={index}
                className={`bg-white rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-all duration-300 transform ${hoveredDish === index ? '-translate-y-2' : ''}`}
                onMouseEnter={() => setHoveredDish(index)}
                onMouseLeave={() => setHoveredDish(null)}
              >
                <div className="h-48 bg-gradient-to-r from-amber-50 to-emerald-50 flex items-center justify-center">
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className="h-20 w-20 text-amber-500" 
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={1.5} 
                      d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" 
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M5 8h14a1 1 0 011 1v.5c0 1.5-2.517 5.573-4 6.5v1a1 1 0 01-1 1H9a1 1 0 01-1-1v-1c-1.483-.927-4-5-4-6.5V9a1 1 0 011-1z"
                    />
                  </svg>
                </div>
                <div className="p-6">
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">{dish.name}</h3>
                  <div className="flex items-center text-gray-600">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1 text-amber-500" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z" />
                    </svg>
                    {dish.calories} kcal per serving
                  </div>
                  {/* <button className="mt-4 w-full py-2 text-sm font-medium text-emerald-600 hover:text-emerald-700 transition-colors">
                    View Details →
                  </button> */}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-emerald-600 to-emerald-700 text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to Explore Bengali Cuisine Smartly?</h2>
          <p className="text-xl mb-8 opacity-90">Join thousands of food enthusiasts who are enjoying their favorite dishes while staying nutritionally informed.</p>
          <a 
            href="/register" 
            className="inline-block px-8 py-4 bg-white text-emerald-700 font-bold rounded-lg shadow-lg hover:bg-gray-100 transition-colors duration-300"
          >
            Get Started for Free
          </a>
        </div>
      </section>
    </div>
  );
}