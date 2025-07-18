import { useState } from "react";
import "./index.css";

function App() {
  // const featureNames = [
  //   "Number of times pregnant",
  //   "Plasma glucose concentration (2 hr OGTT)",
  //   "Diastolic blood pressure (mm Hg)",
  //   "Triceps skin fold thickness (mm)",
  //   "2-Hour serum insulin (μU/ml)",
  //   "Body mass index (kg/m²)",
  //   "Diabetes pedigree function",
  //   "Age (years)",
  // ];

  // const [inputs, setInputs] = useState(Array(featureNames.length).fill(""));
  // const [result, setResult] = useState(null);
  // const [loading, setLoading] = useState(false);
  // const [error, setError] = useState(null);

  // const handleChange = (index, value) => {
  //   const updated = [...inputs];
  //   updated[index] = value;
  //   setInputs(updated);
  // };

  // const handleSubmit = async () => {
  //   setLoading(true);
  //   setError(null);
  //   setResult(null);

  //   try {
  //     const response = await fetch("http://127.0.0.1:5000/predict", {
  //       method: "POST",
  //       headers: {
  //         "Content-Type": "application/json",
  //       },
  //       body: JSON.stringify({
  //         features: inputs.map(Number),
  //       }),
  //     });

  //     if (!response.ok) {
  //       throw new Error("Server error. Please try again.");
  //     }

  //     const data = await response.json();
  //     setResult(data);
  //   } catch (err) {
  //     setError(err.message);
  //   } finally {
  //     setLoading(false);
  //   }
  // };
  // return (
  //   <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 flex items-center justify-center p-6">
  //     <div className="w-full max-w-2xl bg-white shadow-xl rounded-lg p-8">
  //       <h1 className="text-3xl font-extrabold text-center text-blue-800 mb-6">
  //         Diabetes Risk Prediction
  //       </h1>

  //       <p className="text-gray-700 text-center mb-8">
  //         Enter patient health metrics below to estimate diabetes risk.
  //       </p>

  //       <form
  //         onSubmit={(e) => {
  //           e.preventDefault();
  //           handleSubmit();
  //         }}
  //       >
  //         {featureNames.map((name, i) => (
  //           <div key={i} className="mb-4">
  //             <label className="block text-sm font-medium text-gray-700 mb-1">
  //               {name}
  //             </label>
  //             <input
  //               type="number"
  //               step="any"
  //               placeholder={name}
  //               value={inputs[i]}
  //               onChange={(e) => handleChange(i, e.target.value)}
  //               className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
  //               required
  //             />
  //           </div>
  //         ))}

  //         <button
  //           type="submit"
  //           className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded mt-4 transition duration-300"
  //           disabled={loading}
  //         >
  //           {loading ? "Predicting..." : "Predict"}
  //         </button>
  //       </form>

  //       {error && (
  //         <div className="mt-6 bg-red-100 text-red-700 p-4 rounded">
  //           {error}
  //         </div>
  //       )}

  //       {result && (
  //         <div className="mt-8 border-t pt-6">
  //           <h2 className="text-xl font-bold text-gray-800 mb-2">
  //             Prediction Result
  //           </h2>
  //           <p className="text-gray-700">
  //             <span className="font-semibold">Outcome:</span>{" "}
  //             <span
  //               className={`font-bold ${
  //                 result.prediction === 1 ? "text-red-600" : "text-green-600"
  //               }`}
  //             >
  //               {result.prediction === 1
  //                 ? "High Diabetes Risk"
  //                 : "Low Diabetes Risk"}
  //             </span>
  //           </p>

  //           <p className="mt-2 text-gray-700">
  //             <span className="font-semibold">Probabilities:</span>
  //             <br />
  //             <span className="text-gray-600">
  //               Class 0 (No Diabetes):{" "}
  //               <span className="font-mono">
  //                 {Number(result.probability).toFixed(3)}
  //               </span>
  //             </span>
  //             <br />
  //             <span className="text-gray-600">
  //               Class 1 (Diabetes):{" "}
  //               <span className="font-mono">
  //                 {Number(1 - result.probability).toFixed(3)}
  //               </span>
  //             </span>
  //           </p>
  //         </div>
  //       )}
  //     </div>
  //   </div>
  // );

  const [preg, setPreg] = useState(false);
  const [basicData, setBasicData] = useState();
  const [result, setResult] = useState(null);

  const handleBasicSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    setBasicData(data);
    if (data.gender === "Female" && Number(data.age) >= 18) setPreg(true);
  };
  const handlePredict = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const newData = Object.fromEntries(formData.entries());
    const finalData = { ...basicData, ...newData };
    //console.log(finalData);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          finalData,
        }),
      });

      const resultData = await response.json();
      console.log("Prediction result:", resultData);
      setResult(resultData);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex flex-col justify-center items-center mt-20">
      <h1 className="text-xl font-bold text-center mb-3">
        Diabetes Prediction Algorithm
      </h1>

      {!basicData && (
        <form onSubmit={handleBasicSubmit}>
          <div className="flex flex-col justify-center items-center p-4 space-y-5">
            <div className="flex flex-col space-y-5">
              <div className="flex flex-row items-center gap-4">
                <label htmlFor="name" className="w-40">
                  Name
                </label>
                <input
                  type="text"
                  id="name"
                  className="px-3 py-2 border border-gray-300 rounded bg-gray-200"
                />
              </div>

              <div className="flex flex-row items-center gap-4">
                <label htmlFor="age" className="w-40">
                  Age
                </label>
                <input
                  type="number"
                  name="age"
                  id="age"
                  min={0}
                  className="px-3 py-2 border border-gray-300 rounded bg-gray-200"
                />
              </div>

              <div className="flex flex-row items-center gap-4">
                <label htmlFor="gender" className="w-40">
                  Gender:
                </label>
                <select
                  id="gender"
                  name="gender"
                  className="px-3 py-2 border border-gray-300 rounded w-55 bg-gray-200 text-black"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Submit
              </button>
            </div>
          </div>
        </form>
      )}

      {basicData && (
        <form onSubmit={handlePredict}>
          <div className="flex flex-col justify-center items-center p-4 space-y-5">
            <div className="flex flex-row items-center gap-4">
              <label htmlFor="glucose" className="w-40">
                Glucose
              </label>
              <input
                type="number"
                name="glucose"
                id="glucose"
                min={0}
                className="px-3 py-2 border border-gray-300 rounded bg-gray-200"
              />
            </div>

            <div className="flex flex-row items-center gap-4">
              <label htmlFor="bloodPressure" className="w-40">
                Blood Pressure
              </label>
              <input
                type="number"
                name="bloodPressure"
                id="bloodPressure"
                min={0}
                className="px-3 py-2 border border-gray-300 rounded bg-gray-200"
              />
            </div>

            <div className="flex flex-row items-center gap-4">
              <label htmlFor="skinThickness" className="w-40">
                Skin Thickness
              </label>
              <input
                type="number"
                name="skinThickness"
                id="skinThickness"
                min={0}
                className="px-3 py-2 border border-gray-300 rounded bg-gray-200"
              />
            </div>

            <div className="flex flex-row items-center gap-4">
              <label htmlFor="insulin" className="w-40">
                Insulin
              </label>
              <input
                type="number"
                name="insulin"
                id="insulin"
                min={0}
                className="px-3 py-2 border border-gray-300 rounded bg-gray-200"
              />
            </div>

            <div className="flex flex-row items-center gap-4">
              <label htmlFor="bmi" className="w-40">
                BMI
              </label>
              <input
                type="number"
                name="bmi"
                id="bmi"
                min={0}
                className="px-3 py-2 border border-gray-300 rounded bg-gray-200"
              />
            </div>
            {preg && (
              <div className="flex flex-row items-center gap-4">
                <label htmlFor="pregnancies" className="w-40">
                  Pregnancies
                </label>
                <input
                  type="number"
                  name="pregnancies"
                  id="pregnancies"
                  min={0}
                  className="px-3 py-2 border border-gray-300 rounded bg-gray-200"
                />
              </div>
            )}
            <button
              type="submit"
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Predict
            </button>
          </div>
        </form>
      )}
      {result && (
        <div className="mt-8 border-t pt-6 px-4">
          <h2 className="text-xl font-bold text-gray-100 mb-2">
            Prediction Result
          </h2>
          <p className="text-gray-700">
            <span className="font-semibold text-gray-200">Outcome:</span>{" "}
            <span
              className={`font-bold ${
                result.prediction === 1 ? "text-red-600" : "text-green-600"
              }`}
            >
              {result.prediction === 1
                ? "High Diabetes Risk"
                : "Low Diabetes Risk"}
            </span>
          </p>

          <p className="mt-2 text-gray-200">
            <span className="font-semibold">Probabilities:</span>
            <br />
            <span className="text-gray-200">
              Class 0 (No Diabetes):{" "}
              <span className="font-mono">
                {Number(1 - result.probability).toFixed(3)}
              </span>
            </span>
            <br />
            <span className="text-gray-200">
              Class 1 (Diabetes):{" "}
              <span className="font-mono">
                {Number(result.probability).toFixed(3)}
              </span>
            </span>
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
