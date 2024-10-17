import { DefaultNode, Graph } from "@visx/network";

export type NetworkProps = {
  width: number;
  height: number;
};

interface CustomNode {
  x: number;
  y: number;
  size?: number;
  type: string;
}
interface CustomLink {
  source: CustomNode;
  target: CustomNode;
  weight: number;
}

export const background = "#FFFFFF";
export const typeColors: Record<string, string> = {
  input: "red",
  hidden: "blue",
  output: "green",
};

export default function Plot({ width, height }: NetworkProps) {
  const nodeSize = Math.min(width, height) * 0.03;
  const scaleX = (x: number) => (x / 100) * width;
  const scaleY = (y: number) => (y / 100) * height;
  const calculateY = (layerSize: number, index: number) =>
    scaleY((100 * (index + 0.5)) / (layerSize + 0));

  const inputSize = 2;
  const hiddenSize = 3;
  const hiddenLayers = 2;
  const outputSize = 1;

  const layerOffset = 60 / (hiddenLayers + 1);

  const nodesInput: CustomNode[] = Array.from(
    { length: inputSize },
    (_, i) => ({
      x: scaleX(20),
      y: calculateY(inputSize, i),
      type: "input",
    })
  );

  const nodesHidden: CustomNode[][] = Array.from(
    { length: hiddenLayers },
    (_, layerIndex) =>
      Array.from({ length: hiddenSize }, (_, i) => ({
        x: scaleX(20 + (layerIndex + 1) * layerOffset),
        y: calculateY(hiddenSize, i),
        type: "hidden",
      }))
  );

  const nodesOutput: CustomNode[] = Array.from(
    { length: outputSize },
    (_, i) => ({
      x: scaleX(20 + 3 * layerOffset),
      y: calculateY(outputSize, i),
      type: "output",
    })
  );

  const nodes: CustomNode[] = [
    ...nodesInput,
    ...nodesHidden.flat(),
    ...nodesOutput,
  ];

  const links: CustomLink[] = [
    ...nodesInput.flatMap((inputNode) =>
      nodesHidden[0].map((hiddenNode) => ({
        source: inputNode,
        target: hiddenNode,
        weight: Math.random() * 2 - 1, // Random weight between -1 and 1
      }))
    ),
    ...nodesHidden[0].flatMap((hiddenNode1) =>
      nodesHidden[1].map((hiddenNode2) => ({
        source: hiddenNode1,
        target: hiddenNode2,
        weight: Math.random() * 2 - 1,
      }))
    ),
    ...nodesHidden[1].flatMap((hiddenNode2) =>
      nodesOutput.map((outputNode) => ({
        source: hiddenNode2,
        target: outputNode,
        weight: Math.random() * 2 - 1,
      }))
    ),
  ];

  const graph = {
    nodes,
    links,
  };
  return width < 10 ? null : (
    <svg width={width} height={height}>
      <rect width={width} height={height} fill={background} />
      <Graph<CustomLink, CustomNode>
        graph={graph}
        top={0}
        left={0}
        nodeComponent={({ node: { type, size = nodeSize } }) => (
          <DefaultNode r={size} fill={typeColors[type]} />
        )}
        linkComponent={({ link: { source, target, weight } }) => (
          <line
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            strokeWidth={Math.abs(weight) * 6}
            stroke={`rgb(
              ${Math.max(0, -weight * 256)}, 
              ${Math.max(0, weight * 256)}, 
              0)`}
          />
        )}
      />
    </svg>
  );
}
