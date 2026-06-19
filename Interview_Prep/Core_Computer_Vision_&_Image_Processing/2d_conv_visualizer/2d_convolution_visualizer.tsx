import React, { useState, useEffect } from 'react';

export default function App() {
  // 5x5 Input Image (A bright vertical bar in the middle)
  const inputImage = [
    [0, 10, 10, 10, 0],
    [0, 10, 10, 10, 0],
    [0, 10, 10, 10, 0],
    [0, 10, 10, 10, 0],
    [0, 10, 10, 10, 0],
  ];

  // 3x3 Vertical Edge Kernel (Sobel-like)
  const kernel = [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
  ];

  const inputRows = inputImage.length;
  const inputCols = inputImage[0].length;
  const kernelSize = kernel.length;
  
  // No padding, stride 1 -> Output is (W - K + 1)
  const outputRows = inputRows - kernelSize + 1;
  const outputCols = inputCols - kernelSize + 1;
  const maxSteps = outputRows * outputCols - 1;

  const [step, setStep] = useState(0);

  // Calculate current Top-Left position of the kernel on the input image
  const currentRow = Math.floor(step / outputCols);
  const currentCol = step % outputCols;

  // Calculate the entire output matrix for display
  const outputMatrix = Array(outputRows).fill(0).map(() => Array(outputCols).fill(0));
  for (let r = 0; r < outputRows; r++) {
    for (let c = 0; c < outputCols; c++) {
      let sum = 0;
      for (let kr = 0; kr < kernelSize; kr++) {
        for (let kc = 0; kc < kernelSize; kc++) {
          sum += inputImage[r + kr][c + kc] * kernel[kr][kc];
        }
      }
      outputMatrix[r][c] = sum;
    }
  }

  // Generate the math equation for the current step
  const equationParts = [];
  let currentSum = 0;
  for (let kr = 0; kr < kernelSize; kr++) {
    for (let kc = 0; kc < kernelSize; kc++) {
      const imgVal = inputImage[currentRow + kr][currentCol + kc];
      const kernVal = kernel[kr][kc];
      const product = imgVal * kernVal;
      currentSum += product;
      equationParts.push(`(${imgVal} × ${kernVal})`);
    }
  }

  const handleSliderChange = (e) => {
    setStep(parseInt(e.target.value));
  };

  const nextStep = () => setStep((s) => Math.min(maxSteps, s + 1));
  const prevStep = () => setStep((s) => Math.max(0, s - 1));

  // Helper to determine cell background color based on value intensity
  const getCellColor = (val, maxVal = 10, isKernel = false) => {
    if (isKernel) {
      if (val === -1) return 'bg-red-200 text-red-800';
      if (val === 1) return 'bg-green-200 text-green-800';
      return 'bg-slate-100 text-slate-400';
    }
    const intensity = Math.abs(val) / maxVal;
    if (val === 0) return 'bg-slate-50 text-slate-400';
    if (val > 0) return `bg-blue-500 text-white`;
    return `bg-red-500 text-white`;
  };

  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans text-slate-800">
      <div className="max-w-5xl mx-auto space-y-8">
        
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
          <h1 className="text-2xl font-bold mb-2">2D Convolution (Cross-Correlation)</h1>
          <p className="text-slate-600">
            Watch the 3x3 Vertical Edge Kernel slide across the 5x5 Input Image. 
            Notice how it outputs positive numbers for the left edge (black to white) and negative numbers for the right edge (white to black).
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* INPUT IMAGE */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center">
            <h2 className="text-lg font-semibold mb-4 text-slate-700">Input Image (5x5)</h2>
            <div className="grid grid-cols-5 gap-1 bg-slate-200 p-1 rounded-lg">
              {inputImage.map((row, rIdx) => (
                row.map((val, cIdx) => {
                  const isHighlighted = rIdx >= currentRow && rIdx < currentRow + kernelSize &&
                                        cIdx >= currentCol && cIdx < currentCol + kernelSize;
                  return (
                    <div 
                      key={`in-${rIdx}-${cIdx}`}
                      className={`w-10 h-10 flex items-center justify-center rounded font-medium transition-colors duration-200 ${
                        isHighlighted ? 'ring-4 ring-blue-400 ring-inset shadow-lg z-10 ' + getCellColor(val) : getCellColor(val) + ' opacity-60'
                      }`}
                    >
                      {val}
                    </div>
                  );
                })
              ))}
            </div>
          </div>

          {/* KERNEL */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center justify-center">
            <h2 className="text-lg font-semibold mb-4 text-slate-700">Kernel (3x3)</h2>
            <div className="grid grid-cols-3 gap-1 bg-slate-200 p-1 rounded-lg">
              {kernel.map((row, rIdx) => (
                row.map((val, cIdx) => (
                  <div 
                    key={`k-${rIdx}-${cIdx}`}
                    className={`w-12 h-12 flex items-center justify-center rounded font-bold text-lg ${getCellColor(val, 1, true)}`}
                  >
                    {val}
                  </div>
                ))
              ))}
            </div>
            <p className="text-sm text-slate-500 mt-4 text-center">Vertical Edge Detector</p>
          </div>

          {/* OUTPUT IMAGE */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center">
            <h2 className="text-lg font-semibold mb-4 text-slate-700">Output Image (3x3)</h2>
            <div className="grid grid-cols-3 gap-1 bg-slate-200 p-1 rounded-lg relative">
              {outputMatrix.map((row, rIdx) => (
                row.map((val, cIdx) => {
                  // Only show values up to the current step
                  const cellStep = rIdx * outputCols + cIdx;
                  const isVisible = cellStep <= step;
                  const isActive = cellStep === step;
                  
                  return (
                    <div 
                      key={`out-${rIdx}-${cIdx}`}
                      className={`w-14 h-14 flex items-center justify-center rounded font-bold text-lg transition-all duration-300 ${
                        isActive ? 'ring-4 ring-green-400 ring-inset shadow-lg transform scale-110 z-10 ' : ''
                      } ${isVisible ? getCellColor(val, 30) : 'bg-slate-100 text-transparent'}`}
                    >
                      {isVisible ? val : ''}
                    </div>
                  );
                })
              ))}
            </div>
          </div>

        </div>

        {/* MATH BREAKDOWN */}
        <div className="bg-slate-800 text-slate-100 p-6 rounded-2xl shadow-sm">
          <h2 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Step {step + 1} / {maxSteps + 1} Calculation</h2>
          <div className="font-mono text-sm leading-relaxed overflow-x-auto pb-2">
            <span className="text-blue-300">Sum</span> = {equationParts.join(' + ')}
          </div>
          <div className="mt-4 text-2xl font-mono font-bold text-green-400">
            Output = {currentSum}
          </div>
        </div>

        {/* CONTROLS */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col sm:flex-row items-center gap-6">
          <button 
            onClick={prevStep}
            disabled={step === 0}
            className="px-6 py-3 bg-slate-100 text-slate-700 font-semibold rounded-xl disabled:opacity-50 hover:bg-slate-200 transition-colors"
          >
            ← Previous
          </button>
          
          <div className="flex-1 w-full flex items-center gap-4">
            <span className="text-sm font-medium text-slate-500">Start</span>
            <input 
              type="range" 
              min="0" 
              max={maxSteps} 
              value={step} 
              onChange={handleSliderChange}
              className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
            <span className="text-sm font-medium text-slate-500">End</span>
          </div>

          <button 
            onClick={nextStep}
            disabled={step === maxSteps}
            className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-xl disabled:opacity-50 hover:bg-blue-700 transition-colors shadow-md shadow-blue-200"
          >
            Next Step →
          </button>
        </div>

      </div>
    </div>
  );
}