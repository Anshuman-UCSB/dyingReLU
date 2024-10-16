import Plot from "./plot"
import ParentSize from '@visx/responsive/lib/components/ParentSize';

function App() {
  return (
    <>
      {/* <h1 className="text-4xl font-bold">Hello World</h1> */}
      <ParentSize>{({ width, height }) => <Plot width={width} height={height} />}</ParentSize>
    </>
  )
}

export default App
