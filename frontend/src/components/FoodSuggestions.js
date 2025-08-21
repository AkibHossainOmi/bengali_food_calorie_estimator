import React, { useEffect, useState } from "react";

// Food map with categories and allowed meals
const food_map = {
  "Biriyani": { calories: 350, category: "rice", meals: ["lunch", "dinner"] },
  "Khichuri": { calories: 180, category: "rice", meals: ["lunch", "dinner"] },
  "Plain_Rice": { calories: 130, category: "rice", meals: ["lunch", "dinner"] },
  "Boiled_egg": { calories: 77, category: "protein", meals: ["breakfast", "lunch", "dinner"] },
  "Omelette": { calories: 154, category: "protein", meals: ["breakfast", "lunch"] },
  "Hilsha_Fish_Curry": { calories: 400, category: "protein", meals: ["lunch", "dinner"] },
  "Prawn_Curry": { calories: 220, category: "protein", meals: ["lunch", "dinner"] },
  "Chicken_Curry": { calories: 250, category: "protein", meals: ["lunch", "dinner"] },
  "Shutki_Fish": { calories: 310, category: "protein", meals: ["lunch", "dinner"] },
  "Vegetable_Curry": { calories: 90, category: "vegetable", meals: ["lunch", "dinner"] },
  "Mixed_Vegetables": { calories: 80, category: "vegetable", meals: ["lunch", "dinner"] },
  "Salad": { calories: 60, category: "vegetable", meals: ["breakfast", "lunch", "dinner"] },
  "Sandwich": { calories: 250, category: "carb", meals: ["breakfast", "lunch"] },
  "Cake": { calories: 290, category: "dessert", meals: ["breakfast", "snack"] },
  "Chocolate_Cake": { calories: 370, category: "dessert", meals: ["snack"] },
  "Cheesecake": { calories: 321, category: "dessert", meals: ["snack"] },
  "Ice_Cream": { calories: 207, category: "dessert", meals: ["snack", "dinner"] },
  "French_Fries": { calories: 312, category: "side", meals: ["lunch", "snack"] },
  "Fried_Fish": { calories: 280, category: "protein", meals: ["lunch", "dinner"] },
  "Fried_Rice": { calories: 333, category: "rice", meals: ["lunch", "dinner"] },
  "Kacchi_Biriyani": { calories: 360, category: "rice", meals: ["lunch", "dinner"] },
  "Kebab": { calories: 330, category: "protein", meals: ["lunch", "dinner"] },
  "Shik_Kebab": { calories: 300, category: "protein", meals: ["lunch", "dinner"] },
  "Singara": { calories: 220, category: "snack", meals: ["breakfast", "snack"] },
  "Vegetable_Beguni": { calories: 180, category: "snack", meals: ["snack"] },
  "Lentil_Soup": { calories: 116, category: "protein", meals: ["breakfast", "lunch", "dinner"] },
  "Dal_Puri": { calories: 190, category: "carb", meals: ["breakfast", "lunch"] },
  "Poached_Egg": { calories: 72, category: "protein", meals: ["breakfast"] },
  "Pizza": { calories: 285, category: "carb", meals: ["lunch", "dinner"] },
  "Misti": { calories: 250, category: "dessert", meals: ["snack", "dinner"] },
  "Steak": { calories: 271, category: "protein", meals: ["lunch", "dinner"] },
  "Vorta": { calories: 160, category: "side", meals: ["lunch", "dinner"] },
  "Chow_Mein": { calories: 200, category: "carb", meals: ["lunch", "dinner"] },
  "Crab_Dish_Kakra": { calories: 220, category: "protein", meals: ["lunch", "dinner"] },
  "Cup_Cakes": { calories: 270, category: "dessert", meals: ["snack"] },
};

// Helper to randomly pick items from an array
const pickRandom = (arr, count = 1) => {
  const shuffled = [...arr].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, count);
};

// Generate meal suggestions
const generateMeal = (meal) => {
  const eligible = Object.entries(food_map).filter(([name, info]) => info.meals.includes(meal));
  const suggestions = [];

  // Ensure staple for lunch/dinner (rice)
  if (meal === "lunch" || meal === "dinner") {
    const rice = eligible.filter(([_, info]) => info.category === "rice");
    if (rice.length) suggestions.push(rice[Math.floor(Math.random() * rice.length)][0]);
  }

  // Add protein
  const protein = eligible.filter(([_, info]) => info.category === "protein");
  if (protein.length) suggestions.push(protein[Math.floor(Math.random() * protein.length)][0]);

  // Add vegetable or side
  const veg = eligible.filter(([_, info]) => ["vegetable", "side"].includes(info.category));
  if (veg.length) suggestions.push(veg[Math.floor(Math.random() * veg.length)][0]);

  // Optional dessert/snack
  const dessert = eligible.filter(([_, info]) => info.category === "dessert");
  if (dessert.length && Math.random() > 0.5) suggestions.push(dessert[Math.floor(Math.random() * dessert.length)][0]);

  return suggestions; // ALWAYS an array of food names (strings)
};

// Portion-adjusted calories (per 100g)
const getCalories = (food, portion = 100) => {
  const info = food_map[food];
  return Math.round((info.calories / 100) * portion);
};

export default function FoodSuggestions({ weight, height }) {
  const [bmi, setBmi] = useState(null);
  const [category, setCategory] = useState("");
  const [mealSuggestions, setMealSuggestions] = useState({});

  useEffect(() => {
    if (!weight || !height) return;

    const calcBmi = weight / ((height / 100) ** 2);
    setBmi(calcBmi.toFixed(1));

    let cat = "";
    if (calcBmi < 18.5) cat = "underweight";
    else if (calcBmi < 25) cat = "normal";
    else if (calcBmi < 30) cat = "overweight";
    else cat = "obese";
    setCategory(cat);

    // Generate suggestions for each meal
    const meals = ["breakfast", "lunch", "dinner"];
    const suggestions = {};
    meals.forEach(meal => {
      suggestions[meal] = generateMeal(meal).map(food => ({
        name: food.replace(/_/g, " "),
        calories: getCalories(food),
      }));
    });
    setMealSuggestions(suggestions);

  }, [weight, height]);

  if (!weight || !height) {
    return (
      <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100 mt-6">
        <div className="flex items-center justify-center py-8 text-center">
          <div className="max-w-xs">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-gray-300 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-gray-500 font-medium">Enter your weight and height to see personalized food suggestions.</p>
          </div>
        </div>
      </div>
    );
  }

  // Get BMI status color
  const getBmiColor = () => {
    if (category === "underweight") return "text-yellow-600";
    if (category === "normal") return "text-emerald-600";
    if (category === "overweight") return "text-orange-600";
    return "text-red-600";
  };

  return (
    <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100 mt-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-800">Personalized Meal Suggestions</h3>
        <div className="flex items-center bg-gray-50 rounded-full py-1 px-3">
          <span className="text-sm text-gray-600 mr-2">Based on your BMI:</span>
          <span className={`text-sm font-medium ${getBmiColor()}`}>{bmi} ({category})</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {["breakfast", "lunch", "dinner"].map(meal => (
          <div key={meal} className="bg-gray-50 rounded-lg p-4 border border-gray-100">
            <div className="flex items-center mb-4">
              <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center mr-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h4 className="text-lg font-medium text-gray-800 capitalize">{meal}</h4>
            </div>
            
            <div className="space-y-3">
              {mealSuggestions[meal]?.length > 0 ? (
                mealSuggestions[meal].map((item, index) => (
                  <div key={index} className="bg-white rounded-lg p-3 border border-gray-200 shadow-xs flex justify-between items-center">
                    <div>
                      <p className="font-medium text-gray-800 text-sm">{item.name}</p>
                      <p className="text-xs text-gray-500 mt-1">Approx. calories</p>
                    </div>
                    <div className="bg-emerald-50 rounded-md py-1 px-2">
                      <p className="text-emerald-700 font-medium text-sm">{item.calories} kcal</p>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-4 text-gray-400">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                  <p className="text-sm">No suggestions available</p>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-6 pt-5 border-t border-gray-100">
        <div className="flex items-start">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400 mr-2 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-sm text-gray-500">
            These suggestions are based on your BMI and are meant as general guidance. 
            Individual nutritional needs may vary.
          </p>
        </div>
      </div>
    </div>
  );
}