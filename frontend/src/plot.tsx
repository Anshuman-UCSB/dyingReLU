import { DefaultNode, Graph } from '@visx/network';

export type NetworkProps = {
  width: number;
  height: number;
};

interface CustomNode {
  x: number;
  y: number;
  color?: string;
  size?: number;
}

interface CustomLink {
  source: CustomNode;
  target: CustomNode;
  weight: number;
}


export const background = '#FFFFFF';

export default function Plot({ width, height }: NetworkProps) {
    const nodes: CustomNode[] = [
      { x: Math.min(width, height) * .2, y: Math.min(width, height) * .2, size: Math.min(width, height) * .03 },
      { x: Math.min(width, height) * .2, y: Math.min(width, height) * .8, size: Math.min(width, height) * .03 },
      { x: Math.min(width, height) * .8, y: Math.min(width, height) * .5, size: Math.min(width, height) * .03 },
    ];
    
    const links: CustomLink[] = [
      { source: nodes[0], target: nodes[1], weight: .5},
      { source: nodes[1], target: nodes[2], weight: -1.5 },
      { source: nodes[2], target: nodes[0], weight: 1.5 },
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
        nodeComponent={({ node: { color, size } }) =>
          color ? <DefaultNode fill={color} r={size} /> : <DefaultNode r={size} />
        }
        linkComponent={({ link: { source, target, weight } }) => (
          <line
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            strokeWidth={Math.abs(weight)}
            stroke={`rgb(${Math.max(50, -weight * 100)}, ${Math.max(50, weight * 100)}, 0)`}
          />
        )}
      />
    </svg>
  );
}
