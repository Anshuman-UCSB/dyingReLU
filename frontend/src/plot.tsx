import { DefaultNode, Graph } from "@visx/network";
import { Text } from "@visx/text";

export type NetworkProps = {
  width: number;
  height: number;
};

interface CustomNode {
  x: number;
  y: number;
  type: string;
  bias?: number;
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
      bias: Math.random() * 2 - 1,
    })
  );

  const nodesHidden: CustomNode[][] = Array.from(
    { length: hiddenLayers },
    (_, layerIndex) =>
      Array.from({ length: hiddenSize }, (_, i) => ({
        x: scaleX(20 + (layerIndex + 1) * layerOffset),
        y: calculateY(hiddenSize, i),
        type: "hidden",
        bias: Math.random() * 2 - 1,
      }))
  );

  const nodesOutput: CustomNode[] = Array.from(
    { length: outputSize },
    (_, i) => ({
      x: scaleX(20 + 3 * layerOffset),
      y: calculateY(outputSize, i),
      type: "output",
      bias: Math.random() * 2 - 1,
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
        weight: Math.random() * 2 - 1,
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
        nodeComponent={({ node: { type, bias } }) => (
          <>
            <DefaultNode r={nodeSize} fill={typeColors[type]} />
            <Text
              x={-width * 0.03}
              y={-height * 0.01}
              textAnchor="end"
              fontSize={nodeSize * 0.8}
              fill="#333"
            >
              {bias && (bias >= 0 ? "+" : "") + bias.toFixed(2)}
            </Text>
          </>
        )}
        linkComponent={({ link: { source, target, weight } }) => (
          <>
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
            <Text
              x={source.x + (target.x - source.x) * 0.25}
              y={source.y + (target.y - source.y) * 0.25}
              textAnchor="middle"
              fontSize={nodeSize * 0.5}
              fill="#333"
              fontWeight="bold"
              stroke="#fff"
              strokeWidth={2}
              paintOrder="stroke"
            >
              {weight.toFixed(2)}
            </Text>
          </>
        )}
      />
    </svg>
  );
}
