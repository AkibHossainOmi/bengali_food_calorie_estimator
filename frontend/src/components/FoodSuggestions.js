import React, { useEffect, useState } from "react";

// Food map with treats marked
const food_map = {
    Biriyani: { calories: 290, category: "rice", meals: ["lunch", "dinner"], treat: true },
    Khichuri: { calories: 150, category: "rice", meals: ["lunch", "dinner"] },
    Plain_Rice: { calories: 130, category: "rice", meals: ["lunch", "dinner"] },
    Boiled_egg: { calories: 155, category: "protein", meals: ["breakfast", "lunch", "dinner"] },
    Omelette: { calories: 220, category: "protein", meals: ["breakfast", "lunch"] },
    Hilsha_Fish_Curry: { calories: 290, category: "protein", meals: ["lunch", "dinner"], treat: true },
    Prawn_Curry: { calories: 150, category: "protein", meals: ["lunch", "dinner"] },
    Chicken_Curry: { calories: 210, category: "protein", meals: ["lunch", "dinner"] },
    Shutki_Fish: { calories: 280, category: "protein", meals: ["lunch", "dinner"] },
    Vegetable_Curry: { calories: 90, category: "vegetable", meals: ["lunch", "dinner"] },
    Mixed_Vegetables: { calories: 80, category: "vegetable", meals: ["lunch", "dinner"] },
    Salad: { calories: 30, category: "vegetable", meals: ["breakfast", "lunch", "dinner"] },
    Cake: { calories: 310, category: "dessert", meals: ["breakfast", "snack"], treat: true },
    Chocolate_Cake: { calories: 370, category: "dessert", meals: ["snack"], treat: true },
    Cheesecake: { calories: 321, category: "dessert", meals: ["snack"], treat: true },
    Ice_Cream: { calories: 207, category: "dessert", meals: ["snack", "dinner"], treat: true },
    French_Fries: { calories: 312, category: "side", meals: ["lunch", "snack"], treat: true },
    Fried_Fish: { calories: 260, category: "protein", meals: ["lunch", "dinner"] },
    Fried_Rice: { calories: 330, category: "rice", meals: ["lunch", "dinner"], treat: true },
    Kacchi_Biriyani: { calories: 310, category: "rice", meals: ["lunch", "dinner"], treat: true },
    Kebab: { calories: 250, category: "protein", meals: ["lunch", "dinner"], treat: true },
    Shik_Kebab: { calories: 270, category: "protein", meals: ["lunch", "dinner"], treat: true },
    Singara: { calories: 280, category: "snack", meals: ["breakfast", "snack"], treat: true },
    Vegetable_Beguni: { calories: 200, category: "snack", meals: ["snack"], treat: true },
    Lentil_Soup: { calories: 110, category: "protein", meals: ["breakfast", "lunch", "dinner"] },
    Dal_Puri: { calories: 310, category: "carb", meals: ["breakfast", "lunch"], treat: true },
    Poached_Egg: { calories: 143, category: "protein", meals: ["breakfast"] },
    Pizza: { calories: 270, category: "carb", meals: ["lunch", "dinner"], treat: true },
    Misti: { calories: 310, category: "dessert", meals: ["snack", "dinner"], treat: true },
    Steak: { calories: 271, category: "protein", meals: ["lunch", "dinner"] },
    Vorta: { calories: 160, category: "side", meals: ["lunch", "dinner"] },
    Chow_Mein: { calories: 200, category: "carb", meals: ["lunch", "dinner"], treat: true },
    Crab_Dish_Kakra: { calories: 150, category: "protein", meals: ["lunch", "dinner"] },
    Cup_Cakes: { calories: 290, category: "dessert", meals: ["snack"], treat: true },
};

// Portion ranges
const portionMap = {
  breakfast: { rice: [80, 120], protein: [50, 100], vegetable: [30, 60], side: [30, 60], dessert: [30, 60], carb: [50, 100], snack: [30, 60] },
  lunch: { rice: [80, 150], protein: [80, 130], vegetable: [50, 100], side: [50, 80], dessert: [0, 30], carb: [50, 100], snack: [0, 30] },
  dinner: { rice: [80, 150], protein: [80, 130], vegetable: [50, 80], side: [40, 80], dessert: [0, 30], carb: [50, 100], snack: [0, 30] },
};

// Adjust portions based on BMI category
const getPortionByBmi = (bmiCategory, foodCategory, meal) => {
  const [min, max] = portionMap[meal]?.[foodCategory] || [50, 100];
  switch (bmiCategory) {
    case "underweight":
      return Math.floor(max * 1.1);
    case "normal":
      return Math.floor((min + max) / 2);
    case "overweight":
      return Math.floor((min + max) / 2 * 0.9);
    case "obese":
      if (["rice", "carb", "dessert", "snack"].includes(foodCategory)) return Math.floor(min * 0.8);
      if (foodCategory === "protein") return Math.floor((min + max) / 2);
      if (["vegetable", "side"].includes(foodCategory)) return Math.floor(max * 1.1);
      return Math.floor((min + max) / 2);
    default:
      return Math.floor((min + max) / 2);
  }
};

// Generate meal to match target calories
const generateMeal = (meal, bmiCategory, mealCalorieTarget) => {
  let eligible = Object.entries(food_map).filter(([_, info]) => info.meals.includes(meal));

  if (bmiCategory === "obese") eligible = eligible.filter(([_, info]) => !info.treat);
  else if (bmiCategory === "overweight") eligible = eligible.filter(([_, info]) => !info.treat || Math.random() < 0.2);

  const suggestions = [];

  // Always include protein
  const protein = eligible.filter(([_, info]) => info.category === "protein");
  if (protein.length) suggestions.push(protein[Math.floor(Math.random() * protein.length)]);

  // Rice/staple for lunch/dinner
  if (meal === "lunch" || meal === "dinner") {
    const rice = eligible.filter(([_, info]) => info.category === "rice");
    if (rice.length) suggestions.push(rice[Math.floor(Math.random() * rice.length)]);
  }

  // Vegetable/side
  const veg = eligible.filter(([_, info]) => ["vegetable", "side"].includes(info.category));
  if (veg.length) suggestions.push(veg[Math.floor(Math.random() * veg.length)]);

  // Optional dessert/snack
  const dessert = eligible.filter(([_, info]) => ["dessert", "snack"].includes(info.category));
  if (dessert.length && Math.random() > 0.7) suggestions.push(dessert[Math.floor(Math.random() * dessert.length)]);

  // Assign portions to match calorie target
  let totalBaseCalories = suggestions.reduce((sum, [_, info]) => sum + info.calories, 0);
  const mealItems = suggestions.map(([name, info]) => {
    let portion = getPortionByBmi(bmiCategory, info.category, meal);
    let cal = Math.round((info.calories / 100) * portion);
    return { name: name.replace(/_/g, " "), portion, baseCalories: cal, category: info.category };
  });

  const totalMealCalories = mealItems.reduce((sum, i) => sum + i.baseCalories, 0);
  const adjustmentFactor = mealCalorieTarget / totalMealCalories;

  return mealItems.map(item => ({
    ...item,
    portion: Math.round(item.portion * adjustmentFactor),
    calories: Math.round(item.baseCalories * adjustmentFactor),
  }));
};

// Component
export default function FoodSuggestions({ weight, height, dailyCalorieGoal }) {
  const [bmi, setBmi] = useState(null);
  const [category, setCategory] = useState("");
  const [mealSuggestions, setMealSuggestions] = useState({});

  useEffect(() => {
    if (!weight || !height || !dailyCalorieGoal) return;

    const calcBmi = weight / ((height / 100) ** 2);
    setBmi(calcBmi.toFixed(1));

    let cat = "";
    if (calcBmi < 18.5) cat = "underweight";
    else if (calcBmi < 25) cat = "normal";
    else if (calcBmi < 30) cat = "overweight";
    else cat = "obese";
    setCategory(cat);

    // Calorie distribution
    const mealPercents = { breakfast: 0.25, lunch: 0.4, dinner: 0.35 };
    const meals = ["breakfast", "lunch", "dinner"];
    const suggestions = {};
    meals.forEach(meal => {
      const target = dailyCalorieGoal * mealPercents[meal];
      suggestions[meal] = generateMeal(meal, cat, target);
    });
    setMealSuggestions(suggestions);
  }, [weight, height, dailyCalorieGoal]);

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
            <h4 className="text-lg font-medium text-gray-800 capitalize mb-2">{meal}</h4>
            <div className="space-y-3">
              {mealSuggestions[meal]?.length > 0 ? mealSuggestions[meal].map((item, idx) => (
                <div key={idx} className="bg-white rounded-lg p-3 border border-gray-200 shadow-xs flex justify-between items-center">
                  <div>
                    <p className="font-medium text-gray-800 text-sm">{item.name} ({item.portion} g)</p>
                    <p className="text-xs text-gray-500 mt-1">Approx. calories</p>
                  </div>
                  <div className="bg-emerald-50 rounded-md py-1 px-2">
                    <p className="text-emerald-700 font-medium text-sm">{item.calories} kcal</p>
                  </div>
                </div>
              )) : <p className="text-gray-400 text-sm text-center">No suggestions available</p>}
            </div>
          </div>
        ))}
      </div>
      <p className="text-gray-500 text-sm mt-6">Suggestions are based on your BMI and your daily calorie goal. Individual needs may vary.</p>
    </div>
  );
}
