"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import {
  AlertCircle,
  ArrowUpRight,
  Loader2,
  RefreshCw,
  SlidersHorizontal,
} from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const DEFAULT_DIMENSIONS = {
  width: 960,
  height: 640,
};

const GRAPH_LIMIT = 150;
const NEIGHBORHOOD_LIMIT = 75;

type NodeType = "method" | "dataset" | "metric" | "task";
type RelationType = "proposes" | "evaluates_on" | "reports" | "compares";

type GraphNodeLink = {
  id: string;
  label: string;
  type: NodeType;
  relation: RelationType;
  weight: number;
};

type GraphEvidenceItem = {
  paper_id: string;
  paper_title?: string | null;
  snippet?: string | null;
  confidence: number;
  relation: RelationType;
};

type GraphNodeData = {
  id: string;
  type: NodeType;
  label: string;
  entity_id: string;
  paper_count: number;
  aliases: string[];
  description?: string | null;
  top_links: GraphNodeLink[];
  evidence: GraphEvidenceItem[];
  metadata?: Record<string, unknown> | null;
};

type GraphNode = {
  data: GraphNodeData;
};

type GraphEdgeData = {
  id: string;
  source: string;
  target: string;
  type: RelationType;
  weight: number;
  paper_count: number;
  average_confidence: number;
  metadata?: Record<string, unknown> | null;
};

type GraphEdge = {
  data: GraphEdgeData;
};

type GraphMeta = {
  limit: number;
  node_count: number;
  edge_count: number;
  concept_count?: number | null;
  paper_count?: number | null;
  has_more?: boolean | null;
  center_id?: string | null;
  center_type?: NodeType | null;
  filters?: {
    types?: string[];
    relations?: string[];
    min_conf?: number;
  } | null;
};

type GraphResponse = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  meta: GraphMeta;
};

type GraphState = {
  nodes: Record<string, GraphNodeData>;
  edges: Record<string, GraphEdgeData>;
};

type LayoutDimensions = {
  width: number;
  height: number;
};

type LayoutPositions = Record<string, { x: number; y: number }>;

const INITIAL_GRAPH_STATE: GraphState = {
  nodes: {},
  edges: {},
};

const ALL_TYPES: NodeType[] = ["method", "dataset", "metric", "task"];
const ALL_RELATIONS: RelationType[] = ["proposes", "evaluates_on", "reports", "compares"];

const NODE_COLORS: Record<NodeType, string> = {
  method: "#0ea5e9",
  dataset: "#22c55e",
  metric: "#8b5cf6",
  task: "#f97316",
};

const EDGE_COLORS: Record<RelationType, string> = {
  proposes: "#f97316",
  evaluates_on: "#22c55e",
  reports: "#8b5cf6",
  compares: "#64748b",
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const formatMetadataValue = (value: unknown): string => {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "bigint") {
    return String(value);
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  try {
    return JSON.stringify(value);
  } catch (error) {
    return String(value);
  }
};

const formatTypeLabel = (type: NodeType) => {
  switch (type) {
    case "method":
      return "Method";
    case "dataset":
      return "Dataset";
    case "metric":
      return "Metric";
    case "task":
      return "Task";
    default:
      return type;
  }
};

const formatRelationLabel = (relation: RelationType) => {
  switch (relation) {
    case "proposes":
      return "Proposes";
    case "evaluates_on":
      return "Evaluates on";
    case "reports":
      return "Reports";
    case "compares":
      return "Compares";
    default:
      return relation;
  }
};

const computeLayout = (nodes: GraphNodeData[], dimensions: LayoutDimensions): LayoutPositions => {
  const width = Math.max(dimensions.width, DEFAULT_DIMENSIONS.width);
  const height = Math.max(dimensions.height, DEFAULT_DIMENSIONS.height);
  const marginX = Math.max(90, width * 0.08);
  const marginY = Math.max(90, height * 0.08);
  const availableWidth = Math.max(200, width - marginX * 2);
  const availableHeight = Math.max(260, height - marginY * 2);

  const positions: LayoutPositions = {};
  const columnSpacing = ALL_TYPES.length > 1 ? availableWidth / (ALL_TYPES.length - 1) : 0;

  ALL_TYPES.forEach((type, index) => {
    const columnNodes = nodes.filter((node) => node.type === type);
    if (columnNodes.length === 0) {
      return;
    }
    const columnX = marginX + columnSpacing * index;
    const rowSpacing = columnNodes.length > 1 ? Math.min(availableHeight / (columnNodes.length - 1), 180) : 0;

    columnNodes.forEach((node, nodeIndex) => {
      const y = columnNodes.length > 1 ? marginY + rowSpacing * nodeIndex : height / 2;
      positions[node.id] = {
        x: columnX,
        y,
      };
    });
  });

  if (Object.keys(positions).length !== nodes.length) {
    const fallbackRadius = Math.min(width, height) / 3;
    nodes.forEach((node, index) => {
      if (positions[node.id]) {
        return;
      }
      const angle = (2 * Math.PI * index) / nodes.length;
      positions[node.id] = {
        x: width / 2 + fallbackRadius * Math.cos(angle),
        y: height / 2 + fallbackRadius * Math.sin(angle),
      };
    });
  }

  return positions;
};

const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data as unknown;
    if (detail && typeof detail === "object" && "detail" in detail && typeof (detail as any).detail === "string") {
      return (detail as any).detail;
    }
    if (typeof error.message === "string" && error.message.trim().length > 0) {
      return error.message;
    }
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected error while fetching graph data.";
};

const GraphExplorer = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dimensions, setDimensions] = useState<LayoutDimensions>(DEFAULT_DIMENSIONS);
  const [graph, setGraph] = useState<GraphState>(INITIAL_GRAPH_STATE);
  const [meta, setMeta] = useState<GraphMeta | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [isInitialLoading, setIsInitialLoading] = useState<boolean>(true);
  const [expandingNodeId, setExpandingNodeId] = useState<string | null>(null);
  const [selectedTypes, setSelectedTypes] = useState<NodeType[]>(ALL_TYPES);
  const [selectedRelations, setSelectedRelations] = useState<RelationType[]>(ALL_RELATIONS);
  const [minConfidence, setMinConfidence] = useState<number>(0.6);
  const [showEvidence, setShowEvidence] = useState<boolean>(false);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      const nextWidth = Math.max(entry.contentRect.width, DEFAULT_DIMENSIONS.width);
      const nextHeight = Math.max(entry.contentRect.height, DEFAULT_DIMENSIONS.height);
      setDimensions((current) => {
        if (Math.abs(current.width - nextWidth) < 1 && Math.abs(current.height - nextHeight) < 1) {
          return current;
        }
        return { width: nextWidth, height: nextHeight };
      });
    });

    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  const mergeGraphData = useCallback((payload: GraphResponse, options?: { reset?: boolean; allowSelect?: boolean }) => {
    setGraph((previous) => {
      const nextNodes = options?.reset ? {} : { ...previous.nodes };
      payload.nodes.forEach((node) => {
        nextNodes[node.data.id] = {
          ...node.data,
          aliases: node.data.aliases ?? [],
          top_links: node.data.top_links ?? [],
          evidence: node.data.evidence ?? [],
          metadata: node.data.metadata ?? undefined,
        };
      });

      const nextEdges = options?.reset ? {} : { ...previous.edges };
      payload.edges.forEach((edge) => {
        nextEdges[edge.data.id] = {
          ...edge.data,
          metadata: edge.data.metadata ?? undefined,
        };
      });

      return {
        nodes: nextNodes,
        edges: nextEdges,
      };
    });

    setMeta(payload.meta);
    setError(null);

    if (options?.allowSelect) {
      setSelectedNodeId((current) => {
        if (current && payload.nodes.some((node) => node.data.id === current)) {
          return current;
        }
        return payload.meta.center_id ?? payload.nodes[0]?.data.id ?? current;
      });
    }
  }, []);

  const buildFilterParams = useCallback(() => {
    return {
      types: selectedTypes.join(","),
      relations: selectedRelations.join(","),
      min_conf: Number(minConfidence.toFixed(2)),
    };
  }, [minConfidence, selectedRelations, selectedTypes]);

  const loadOverview = useCallback(async () => {
    setIsInitialLoading(true);
    setError(null);
    setExpandedNodes({});
    try {
      const params = { limit: GRAPH_LIMIT, ...buildFilterParams() };
      const response = await axios.get<GraphResponse>(`${API_BASE_URL}/api/graph/overview`, {
        params,
      });
      mergeGraphData(response.data, { reset: true, allowSelect: true });
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsInitialLoading(false);
      setExpandingNodeId(null);
    }
  }, [buildFilterParams, mergeGraphData]);

  useEffect(() => {
    void loadOverview();
  }, [loadOverview]);

  useEffect(() => {
    if (selectedNodeId && !graph.nodes[selectedNodeId]) {
      const fallback = Object.keys(graph.nodes)[0] ?? null;
      setSelectedNodeId(fallback);
    }
  }, [graph.nodes, selectedNodeId]);

  useEffect(() => {
    setShowEvidence(false);
  }, [selectedNodeId]);

  const expandNode = useCallback(
    async (nodeId: string) => {
      const node = graph.nodes[nodeId];
      if (!node) {
        return;
      }
      if (expandedNodes[nodeId]) {
        return;
      }

      const targetId = node.entity_id;
      if (!targetId) {
        return;
      }

      setExpandingNodeId(nodeId);
      try {
        const params = { limit: NEIGHBORHOOD_LIMIT, ...buildFilterParams() };
        const response = await axios.get<GraphResponse>(`${API_BASE_URL}/api/graph/neighborhood/${targetId}`, {
          params,
        });
        mergeGraphData(response.data, { allowSelect: false });
        setExpandedNodes((previous) => ({
          ...previous,
          [nodeId]: true,
        }));
      } catch (err) {
        setError(getErrorMessage(err));
      } finally {
        setExpandingNodeId((current) => (current === nodeId ? null : current));
      }
    },
    [buildFilterParams, expandedNodes, graph.nodes, mergeGraphData]
  );

  const handleNodeSelect = useCallback(
    (nodeId: string) => {
      setSelectedNodeId(nodeId);
      void expandNode(nodeId);
    },
    [expandNode]
  );

  const nodes = useMemo(() => Object.values(graph.nodes), [graph.nodes]);
  const edges = useMemo(() => Object.values(graph.edges), [graph.edges]);

  const positions = useMemo(() => computeLayout(nodes, dimensions), [nodes, dimensions]);

  const nodesWithPositions = useMemo(
    () =>
      nodes.map((node) => ({
        ...node,
        x: positions[node.id]?.x ?? dimensions.width / 2,
        y: positions[node.id]?.y ?? dimensions.height / 2,
      })),
    [nodes, positions, dimensions]
  );

  const selectedNode = selectedNodeId ? graph.nodes[selectedNodeId] : undefined;

  const neighborSet = useMemo(() => {
    if (!selectedNodeId) {
      return new Set<string>();
    }
    const set = new Set<string>();
    edges.forEach((edge) => {
      if (edge.source === selectedNodeId) {
        set.add(edge.target);
      } else if (edge.target === selectedNodeId) {
        set.add(edge.source);
      }
    });
    return set;
  }, [edges, selectedNodeId]);

  const stats = useMemo(() => {
    const methodCount = nodes.filter((node) => node.type === "method").length;
    const datasetCount = nodes.filter((node) => node.type === "dataset").length;
    const metricCount = nodes.filter((node) => node.type === "metric").length;
    const taskCount = nodes.filter((node) => node.type === "task").length;
    return [
      { label: "Total nodes", value: nodes.length },
      { label: "Methods", value: methodCount },
      { label: "Datasets", value: datasetCount },
      { label: "Edges", value: edges.length },
      { label: "Metrics", value: metricCount },
      { label: "Tasks", value: taskCount },
    ];
  }, [edges.length, nodes]);

  const graphSummary = useMemo(() => {
    if (!meta) {
      return null;
    }
    const pieces: string[] = [];
    if (meta.paper_count) {
      pieces.push(`${meta.paper_count} unique papers represented`);
    }
    if (meta.filters) {
      const types = meta.filters.types?.join(", ");
      const relations = meta.filters.relations?.join(", ");
      if (types) {
        pieces.push(`Types: ${types}`);
      }
      if (relations) {
        pieces.push(`Relations: ${relations}`);
      }
      if (typeof meta.filters.min_conf === "number") {
        pieces.push(`Min confidence: ${(meta.filters.min_conf * 100).toFixed(0)}%`);
      }
    }
    if (meta.has_more) {
      pieces.push("Additional results available — expand nodes to load more");
    }
    return pieces.join(" • ");
  }, [meta]);

  const handleRefresh = () => {
    void loadOverview();
  };

  const handleExpandSelected = () => {
    if (selectedNodeId) {
      void expandNode(selectedNodeId);
    }
  };

  const toggleType = (type: NodeType) => {
    setSelectedTypes((current) => {
      if (current.includes(type)) {
        if (current.length === 1) {
          return current;
        }
        return current.filter((item) => item !== type);
      }
      const withType = [...current, type];
      return ALL_TYPES.filter((item) => withType.includes(item));
    });
  };

  const toggleRelation = (relation: RelationType) => {
    setSelectedRelations((current) => {
      if (current.includes(relation)) {
        if (current.length === 1) {
          return current;
        }
        return current.filter((item) => item !== relation);
      }
      const withRelation = [...current, relation];
      return ALL_RELATIONS.filter((item) => withRelation.includes(item));
    });
  };

  const handleConfidenceChange = (value: number) => {
    setMinConfidence(clamp(Number(value), 0.5, 1));
  };

  const hasGraphData = nodesWithPositions.length > 0;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
      <div className="space-y-5">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {stats.map((item) => (
            <div key={item.label} className="rounded-lg border bg-card p-4 shadow-sm transition hover:shadow-md">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{item.label}</p>
              <p className="mt-3 text-2xl font-semibold text-foreground">{item.value.toLocaleString()}</p>
            </div>
          ))}
        </div>

        <div className="rounded-lg border bg-card p-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <SlidersHorizontal className="h-4 w-4" />
              Filters
            </div>
            <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                {ALL_TYPES.map((type) => {
                  const isActive = selectedTypes.includes(type);
                  return (
                    <button
                      key={type}
                      type="button"
                      onClick={() => toggleType(type)}
                      className={`inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-medium transition ${
                        isActive
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border bg-background text-muted-foreground hover:bg-muted"
                      }`}
                    >
                      {formatTypeLabel(type)}
                    </button>
                  );
                })}
              </div>
              <div className="flex items-center gap-2">
                <span className="uppercase tracking-wide">Confidence</span>
                <input
                  type="range"
                  min={0.5}
                  max={1}
                  step={0.05}
                  value={minConfidence}
                  onChange={(event) => handleConfidenceChange(Number(event.target.value))}
                  className="h-1 w-24 cursor-pointer"
                />
                <span className="font-semibold text-foreground">{(minConfidence * 100).toFixed(0)}%</span>
              </div>
              <div className="flex items-center gap-2">
                {ALL_RELATIONS.map((relation) => {
                  const isActive = selectedRelations.includes(relation);
                  return (
                    <button
                      key={relation}
                      type="button"
                      onClick={() => toggleRelation(relation)}
                      className={`inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-medium transition ${
                        isActive
                          ? "border-secondary bg-secondary/10 text-secondary-foreground"
                          : "border-border bg-background text-muted-foreground hover:bg-muted"
                      }`}
                    >
                      {formatRelationLabel(relation)}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {error ? (
          <div className="flex items-center gap-3 rounded-md border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
            <AlertCircle className="h-4 w-4" />
            <span>{error}</span>
          </div>
        ) : null}

        <div className="overflow-hidden rounded-lg border bg-card shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b px-4 py-3">
            <div>
              <h2 className="text-lg font-semibold text-foreground">Knowledge graph</h2>
              <p className="text-xs text-muted-foreground">Explore typed research entities and their structured relationships.</p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleRefresh}
                className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 py-1.5 text-sm font-medium text-foreground shadow-sm transition hover:bg-muted"
              >
                <RefreshCw className="h-4 w-4" />
                Reset graph
              </button>
              <button
                type="button"
                onClick={handleExpandSelected}
                disabled={!selectedNodeId || expandingNodeId === selectedNodeId}
                className="inline-flex items-center gap-2 rounded-md border border-primary bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground shadow-sm transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <ArrowUpRight className="h-4 w-4" />
                Expand selected
              </button>
            </div>
          </div>

          <div ref={containerRef} className="relative h-[540px] w-full bg-background">
            {isInitialLoading ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            ) : null}

            {hasGraphData ? (
              <svg width={dimensions.width} height={dimensions.height} className="block">
                <defs>
                  <filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="2" stdDeviation="4" floodOpacity="0.18" />
                  </filter>
                </defs>

                {edges.map((edge) => {
                  const source = graph.nodes[edge.source];
                  const target = graph.nodes[edge.target];
                  if (!source || !target) {
                    return null;
                  }
                  const sourcePos = positions[source.id];
                  const targetPos = positions[target.id];
                  if (!sourcePos || !targetPos) {
                    return null;
                  }
                  const strokeWidth = clamp(1 + Math.log(edge.weight + 1), 1.25, 4.5);
                  const color = EDGE_COLORS[edge.type];
                  const isActive = selectedNodeId && (edge.source === selectedNodeId || edge.target === selectedNodeId);
                  return (
                    <line
                      key={edge.id}
                      x1={sourcePos.x}
                      y1={sourcePos.y}
                      x2={targetPos.x}
                      y2={targetPos.y}
                      stroke={color}
                      strokeWidth={strokeWidth}
                      strokeOpacity={isActive ? 0.75 : 0.35}
                      className="transition-opacity"
                    />
                  );
                })}

                {nodesWithPositions.map((node) => {
                  const isSelected = node.id === selectedNodeId;
                  const isNeighbor = neighborSet.has(node.id);
                  const color = NODE_COLORS[node.type];
                  const radius = isSelected ? 18 : 14;
                  return (
                    <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                      <circle
                        r={radius}
                        fill={color}
                        fillOpacity={isSelected ? 1 : isNeighbor ? 0.9 : 0.75}
                        stroke={isSelected ? "#1f2937" : "#0f172a"}
                        strokeWidth={isSelected ? 3 : 2}
                        filter="url(#node-shadow)"
                        className="cursor-pointer transition-opacity"
                        onClick={() => handleNodeSelect(node.id)}
                      />
                      <text
                        x={0}
                        y={radius + 18}
                        textAnchor="middle"
                        className="select-none font-semibold text-slate-900"
                        style={{ fontSize: "12px" }}
                      >
                        {node.label}
                      </text>
                    </g>
                  );
                })}
              </svg>
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                Adjust filters to load graph data.
              </div>
            )}

            {expandingNodeId ? (
              <div className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2 rounded-full border border-primary/40 bg-background/80 px-4 py-1.5 text-xs font-medium text-primary shadow">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Expanding neighborhood…
              </div>
            ) : null}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4 rounded-lg border bg-card px-4 py-3 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full" style={{ backgroundColor: NODE_COLORS.method }} />
            Method nodes
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full" style={{ backgroundColor: NODE_COLORS.dataset }} />
            Dataset nodes
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full" style={{ backgroundColor: NODE_COLORS.metric }} />
            Metric nodes
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full" style={{ backgroundColor: NODE_COLORS.task }} />
            Task nodes
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex h-3.5 w-8 items-center justify-center rounded-full bg-muted text-[10px] font-semibold uppercase text-muted-foreground">
              Edge
            </span>
            Edge color encodes relation type
          </div>
        </div>

        {graphSummary ? (
          <div className="rounded-lg border border-dashed bg-muted/40 px-4 py-3 text-xs text-muted-foreground">
            {graphSummary}
          </div>
        ) : null}
      </div>

      <aside className="space-y-4">
        <div className="rounded-lg border bg-card p-5 shadow-sm">
          <h3 className="text-base font-semibold text-foreground">Node details</h3>
          {selectedNode ? (
            <div className="mt-4 space-y-4 text-sm">
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-primary">Label</p>
                <p className="mt-1 text-lg font-semibold text-foreground">{selectedNode.label}</p>
              </div>

              <div className="grid gap-2 text-xs text-muted-foreground">
                <div className="flex items-center justify-between">
                  <span className="uppercase tracking-wide">Type</span>
                  <span className="rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 font-semibold text-primary">
                    {formatTypeLabel(selectedNode.type)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="uppercase tracking-wide">Used by</span>
                  <span className="font-semibold text-foreground">{selectedNode.paper_count.toLocaleString()} papers</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="uppercase tracking-wide">Expanded</span>
                  <span className="font-semibold text-foreground">
                    {expandedNodes[selectedNode.id] ? "Yes" : "No"}
                  </span>
                </div>
              </div>

              {selectedNode.aliases && selectedNode.aliases.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Aliases</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedNode.aliases.map((alias) => (
                      <span key={alias} className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
                        {alias}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}

              {selectedNode.metadata && Object.keys(selectedNode.metadata).length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Metadata</p>
                  <dl className="grid gap-1 text-xs text-muted-foreground">
                    {Object.entries(selectedNode.metadata).map(([key, value]) => (
                      <div key={key} className="flex justify-between gap-2">
                        <dt className="font-medium capitalize text-foreground/70">{key}</dt>
                        <dd className="text-right">{formatMetadataValue(value)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ) : null}

              {selectedNode.top_links.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Top linked nodes</p>
                  <ul className="space-y-1 text-xs text-muted-foreground">
                    {selectedNode.top_links.map((link) => (
                      <li
                        key={link.id}
                        className="flex items-center justify-between rounded-md border border-border/60 bg-muted/30 px-2 py-1"
                      >
                        <div className="flex flex-col">
                          <span className="font-semibold text-foreground/80">{link.label}</span>
                          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                            {formatRelationLabel(link.relation)}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="rounded-full bg-background px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                            {formatTypeLabel(link.type)}
                          </span>
                          <div className="text-[10px] font-medium text-muted-foreground">
                            Weight {link.weight.toFixed(2)}
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">Expand the node to reveal its strongest connections.</p>
              )}

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Evidence</p>
                  <button
                    type="button"
                    onClick={() => setShowEvidence((current) => !current)}
                    className="text-xs font-medium text-primary underline-offset-2 hover:underline"
                  >
                    {showEvidence ? "Hide" : "Why?"}
                  </button>
                </div>
                {showEvidence ? (
                  selectedNode.evidence.length > 0 ? (
                    <ul className="space-y-2 text-xs text-muted-foreground">
                      {selectedNode.evidence.map((item, index) => (
                        <li key={`${item.paper_id}-${index}`} className="rounded-md border border-border/60 bg-muted/30 p-2">
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-foreground/80">{item.paper_title ?? "Unknown paper"}</span>
                            <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                              {formatRelationLabel(item.relation)} • {(item.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          {item.snippet ? <p className="mt-1 text-[13px] leading-snug text-foreground/80">“{item.snippet}”</p> : null}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-muted-foreground">Evidence snippets will appear after expanding connected edges.</p>
                  )
                ) : null}
              </div>
            </div>
          ) : (
            <p className="mt-3 text-sm text-muted-foreground">Select a node in the graph to view its metadata and connections.</p>
          )}
        </div>

        <div className="rounded-lg border bg-card p-5 text-sm text-muted-foreground">
          <h3 className="text-base font-semibold text-foreground">How to explore</h3>
          <ul className="mt-3 space-y-2 text-sm">
            <li>Toggle entity types and relation categories to focus on specific portions of the graph.</li>
            <li>Adjust the confidence slider to hide uncertain relationships and highlight stronger evidence.</li>
            <li>Click any node to load its neighborhood and inspect the top supporting papers via the “Why?” button.</li>
          </ul>
        </div>
      </aside>
    </div>
  );
};

export default GraphExplorer;

