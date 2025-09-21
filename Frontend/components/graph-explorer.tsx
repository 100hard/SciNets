"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import {
  AlertCircle,
  ArrowUpRight,
  Loader2,
  RefreshCw,
  Target,
} from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const DEFAULT_DIMENSIONS = {
  width: 960,
  height: 640,
};

const GRAPH_LIMIT = 75;
const NEIGHBORHOOD_LIMIT = 60;

type NodeType = "paper" | "concept";

type GraphNodeData = {
  id: string;
  type: NodeType;
  label: string;
  paper_id?: string | null;
  concept_id?: string | null;
  metadata?: Record<string, unknown> | null;
};

type GraphNode = {
  data: GraphNodeData;
};

type GraphEdgeData = {
  id: string;
  source: string;
  target: string;
  type: string;
  paper_id?: string | null;
  concept_id?: string | null;
  related_concept_id?: string | null;
  relation_id?: string | null;
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
  filters?: Record<string, unknown> | null;
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

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const formatMetadataValue = (value: unknown): string => {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "bigint") {
    return value.toString();
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

const getPaperSubtitle = (metadata?: Record<string, unknown> | null): string => {
  if (!metadata || typeof metadata !== "object") {
    return "";
  }
  const record = metadata as Record<string, unknown>;
  const authors = record["authors"];
  if (typeof authors === "string" && authors.trim().length > 0) {
    return authors;
  }
  const venue = record["venue"];
  if (typeof venue === "string" && venue.trim().length > 0) {
    return venue;
  }
  const year = record["year"];
  if (typeof year === "number" || (typeof year === "string" && year.trim().length > 0)) {
    return String(year);
  }
  return "";
};

const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data as unknown;
    if (detail && typeof detail === "object" && "detail" in detail && typeof detail.detail === "string") {
      return detail.detail;
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

const computeLayout = (nodes: GraphNodeData[], dimensions: LayoutDimensions): LayoutPositions => {
  const width = Math.max(dimensions.width, DEFAULT_DIMENSIONS.width);
  const height = Math.max(dimensions.height, DEFAULT_DIMENSIONS.height);
  const marginX = Math.max(120, width * 0.1);
  const marginY = Math.max(120, height * 0.12);
  const availableWidth = Math.max(200, width - marginX * 2);
  const availableHeight = Math.max(220, height - marginY * 2);

  const papers = nodes.filter((node) => node.type === "paper");
  const concepts = nodes.filter((node) => node.type === "concept");

  const positions: LayoutPositions = {};

  if (papers.length === 0 && concepts.length === 0) {
    return positions;
  }

  const estimatedColumns = papers.length > 0 ? Math.max(1, Math.floor(availableWidth / 240)) : 1;
  const columnCount = papers.length > 0 ? Math.min(papers.length, estimatedColumns) : 1;
  const rowCount = papers.length > 0 ? Math.ceil(papers.length / columnCount) : 1;
  const columnSpacing = columnCount > 1 ? availableWidth / (columnCount - 1) : 0;
  const rowSpacing = rowCount > 1 ? Math.min(availableHeight / (rowCount - 1), 280) : 0;

  papers.forEach((paper, index) => {
    const column = columnCount > 1 ? index % columnCount : 0;
    const row = columnCount > 1 ? Math.floor(index / columnCount) : index;
    const x = columnCount > 1 ? marginX + columnSpacing * column : width / 2;
    const y = rowCount > 1 ? marginY + rowSpacing * row : height / 2;
    positions[paper.id] = { x, y };
  });

  const groupedConcepts = new Map<string, GraphNodeData[]>();
  concepts.forEach((concept) => {
    const paperKey = concept.paper_id ? `paper:${concept.paper_id}` : null;
    const key = paperKey && positions[paperKey] ? paperKey : `orphan:${concept.id}`;
    const group = groupedConcepts.get(key);
    if (group) {
      group.push(concept);
    } else {
      groupedConcepts.set(key, [concept]);
    }
  });

  groupedConcepts.forEach((group, key) => {
    const center = positions[key] ?? { x: width / 2, y: height / 2 };
    const perRing = Math.min(12, group.length);
    const baseRadius = 110;
    const ringGap = 70;

    group.forEach((concept, index) => {
      const ringIndex = perRing > 0 ? Math.floor(index / perRing) : 0;
      const indexWithinRing = perRing > 0 ? index % perRing : index;
      const radius = baseRadius + ringGap * ringIndex;
      const angle = perRing === 1 ? -Math.PI / 2 : (2 * Math.PI * indexWithinRing) / perRing - Math.PI / 2;
      const x = center.x + radius * Math.cos(angle);
      const y = center.y + radius * Math.sin(angle);
      const clampedX = clamp(x, marginX * 0.4, width - marginX * 0.4);
      const clampedY = clamp(y, marginY * 0.4, height - marginY * 0.4);
      positions[concept.id] = { x: clampedX, y: clampedY };
    });
  });

  nodes.forEach((node, index) => {
    if (!positions[node.id]) {
      const gridColumns = Math.max(1, Math.floor(availableWidth / 200));
      const col = index % gridColumns;
      const row = Math.floor(index / gridColumns);
      const x = marginX + (gridColumns > 1 ? (availableWidth / (gridColumns - 1 || 1)) * col : availableWidth / 2);
      const y = marginY + Math.min(availableHeight, row * 160);
      positions[node.id] = { x, y };
    }
  });

  return positions;
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

  const mergeGraphData = useCallback(
    (payload: GraphResponse, options?: { reset?: boolean; allowSelect?: boolean }) => {
      setGraph((previous) => {
        const nextNodes = options?.reset ? {} : { ...previous.nodes };
        payload.nodes.forEach((node) => {
          nextNodes[node.data.id] = {
            ...node.data,
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
    },
    []
  );

  const loadOverview = useCallback(async () => {
    setIsInitialLoading(true);
    setError(null);
    setExpandedNodes({});
    try {
      const response = await axios.get<GraphResponse>(`${API_BASE_URL}/api/graph/overview`, {
        params: { limit: GRAPH_LIMIT },
      });
      mergeGraphData(response.data, { reset: true, allowSelect: true });
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsInitialLoading(false);
      setExpandingNodeId(null);
    }
  }, [mergeGraphData]);

  useEffect(() => {
    void loadOverview();
  }, [loadOverview]);

  useEffect(() => {
    if (selectedNodeId && !graph.nodes[selectedNodeId]) {
      const fallback = Object.keys(graph.nodes)[0] ?? null;
      setSelectedNodeId(fallback);
    }
  }, [graph.nodes, selectedNodeId]);

  const expandNode = useCallback(
    async (nodeId: string) => {
      const node = graph.nodes[nodeId];
      if (!node) {
        return;
      }
      if (expandedNodes[nodeId]) {
        return;
      }

      const targetId = node.type === "paper" ? node.paper_id : node.concept_id ?? node.paper_id;
      if (!targetId) {
        return;
      }

      setExpandingNodeId(nodeId);
      try {
        const response = await axios.get<GraphResponse>(`${API_BASE_URL}/api/graph/neighborhood/${targetId}`, {
          params: { limit: NEIGHBORHOOD_LIMIT },
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
    [expandedNodes, graph.nodes, mergeGraphData]
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

  const selectedMetadataEntries = useMemo(() => {
    if (!selectedNode || !selectedNode.metadata || typeof selectedNode.metadata !== "object") {
      return [] as Array<[string, unknown]>;
    }
    return Object.entries(selectedNode.metadata as Record<string, unknown>);
  }, [selectedNode]);

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

  const connectedNodes = useMemo(() => {
    if (!selectedNodeId) {
      return [] as GraphNodeData[];
    }
    const connections: GraphNodeData[] = [];
    neighborSet.forEach((id) => {
      const node = graph.nodes[id];
      if (node) {
        connections.push(node);
      }
    });
    return connections.sort((a, b) => a.label.localeCompare(b.label));
  }, [graph.nodes, neighborSet, selectedNodeId]);

  const stats = useMemo(
    () => {
      const paperCount = nodes.filter((node) => node.type === "paper").length;
      const conceptCount = nodes.filter((node) => node.type === "concept").length;
      return [
        {
          label: "Total nodes",
          value: nodes.length,
        },
        {
          label: "Concepts",
          value: conceptCount,
        },
        {
          label: "Papers",
          value: paperCount,
        },
        {
          label: "Edges",
          value: edges.length,
        },
      ];
    },
    [edges.length, nodes]
  );

  const graphSummary = useMemo(() => {
    if (!meta) {
      return null;
    }
    const pieces: string[] = [];
    if (meta.filters) {
      const filterEntries = Object.entries(meta.filters);
      if (filterEntries.length > 0) {
        pieces.push(
          `Filters: ${filterEntries
            .map(([key, value]) => `${key}=${formatMetadataValue(value)}`)
            .join(", ")}`
        );
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

  const hasGraphData = nodesWithPositions.length > 0;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {stats.map((item) => (
            <div
              key={item.label}
              className="rounded-lg border bg-card p-4 shadow-sm transition hover:shadow-md"
            >
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{item.label}</p>
              <p className="mt-3 text-2xl font-semibold text-foreground">{item.value.toLocaleString()}</p>
            </div>
          ))}
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
              <p className="text-xs text-muted-foreground">Click nodes to expand their neighborhoods and inspect relationships.</p>
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
                disabled={!selectedNodeId}
                className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground shadow-sm transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:bg-primary/40"
              >
                <Target className="h-4 w-4" />
                Expand selection
              </button>
            </div>
          </div>

          {graphSummary ? (
            <div className="border-b px-4 py-2 text-xs text-muted-foreground">{graphSummary}</div>
          ) : null}

          <div ref={containerRef} className="relative h-[640px] bg-muted/30">
            {isInitialLoading ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-background/70 text-sm font-medium text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
                Loading graph…
              </div>
            ) : null}

            {!isInitialLoading && !hasGraphData ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
                <ArrowUpRight className="h-5 w-5 text-muted-foreground/60" />
                <p>No graph data available yet.</p>
                <p>Use the ingestion pipeline to add papers and concepts.</p>
              </div>
            ) : null}

            {!isInitialLoading && hasGraphData ? (
              <svg
                width={dimensions.width}
                height={dimensions.height}
                viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                className="block"
              >
                <defs>
                  <radialGradient id="nodeGlow" cx="50%" cy="50%" r="65%">
                    <stop offset="0%" stopColor="rgba(99, 102, 241, 0.35)" />
                    <stop offset="100%" stopColor="rgba(99, 102, 241, 0)" />
                  </radialGradient>
                </defs>
                <g strokeLinecap="round" strokeLinejoin="round">
                  {edges.map((edge) => {
                    const source = positions[edge.source];
                    const target = positions[edge.target];
                    if (!source || !target) {
                      return null;
                    }
                    const isActive =
                      edge.source === selectedNodeId ||
                      edge.target === selectedNodeId;
                    const isRelation = edge.type !== "mentions";
                    const stroke = isActive ? "#2563eb" : isRelation ? "#fb923c" : "#94a3b8";
                    const opacity = isActive ? 0.95 : isRelation ? 0.85 : 0.6;
                    const strokeWidth = isActive ? 2.8 : isRelation ? 2.4 : 1.6;
                    return (
                      <line
                        key={edge.id}
                        x1={source.x}
                        y1={source.y}
                        x2={target.x}
                        y2={target.y}
                        stroke={stroke}
                        strokeWidth={strokeWidth}
                        strokeOpacity={opacity}
                      />
                    );
                  })}
                </g>

                {nodesWithPositions.map((node) => {
                  const isSelected = node.id === selectedNodeId;
                  const isNeighbor = neighborSet.has(node.id);
                  const paper = node.type === "paper";
                  const fill = paper ? "#0f172a" : "#0284c7";
                  const stroke = isSelected ? "#6366f1" : paper ? "#1e293b" : "#0ea5e9";
                  const textColor = paper ? "#f8fafc" : "#f0f9ff";
                  const secondaryText = paper ? "#cbd5f5" : "#e0f2fe";
                  const shadowOpacity = isSelected ? 0.8 : isNeighbor ? 0.4 : 0;
                  const subtitle = paper ? getPaperSubtitle(node.metadata) : "";

                  return (
                    <g
                      key={node.id}
                      tabIndex={0}
                      role="button"
                      onClick={() => handleNodeSelect(node.id)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          handleNodeSelect(node.id);
                        }
                      }}
                      className="cursor-pointer focus:outline-none"
                    >
                      {shadowOpacity > 0 ? (
                        <circle
                          cx={node.x}
                          cy={node.y}
                          r={paper ? 48 : 42}
                          fill="url(#nodeGlow)"
                          fillOpacity={shadowOpacity}
                        />
                      ) : null}

                      {paper ? (
                        <rect
                          x={node.x - 90}
                          y={node.y - 40}
                          width={180}
                          height={80}
                          rx={14}
                          ry={14}
                          fill={fill}
                          stroke={stroke}
                          strokeWidth={isSelected ? 3 : 2}
                          opacity={isSelected ? 0.98 : isNeighbor ? 0.92 : 0.85}
                          className="transition"
                        />
                      ) : (
                        <circle
                          cx={node.x}
                          cy={node.y}
                          r={28}
                          fill={fill}
                          stroke={stroke}
                          strokeWidth={isSelected ? 3 : 2}
                          opacity={isSelected ? 0.98 : isNeighbor ? 0.88 : 0.82}
                          className="transition"
                        />
                      )}

                      <text
                        x={node.x}
                        y={paper ? node.y - 6 : node.y + 4}
                        textAnchor="middle"
                        fontSize={paper ? 13 : 12}
                        fontWeight={isSelected ? 600 : 500}
                        fill={textColor}
                        style={{ pointerEvents: "none" }}
                      >
                        {node.label}
                      </text>

                      {paper && subtitle ? (
                        <text
                          x={node.x}
                          y={node.y + 18}
                          textAnchor="middle"
                          fontSize={11}
                          fill={secondaryText}
                          style={{ pointerEvents: "none" }}
                        >
                          {subtitle}
                        </text>
                      ) : null}
                    </g>
                  );
                })}
              </svg>
            ) : null}

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
            <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-sm bg-slate-900" />
            Paper nodes
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full bg-sky-500" />
            Concept nodes
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex h-3.5 w-1 rounded-full bg-slate-400" />
            Mentions
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex h-3.5 w-1 rounded-full bg-amber-400" />
            Concept relations
          </div>
        </div>
      </div>

      <aside className="space-y-4">
        <div className="rounded-lg border bg-card p-5 shadow-sm">
          <h3 className="text-base font-semibold text-foreground">Node details</h3>
          {selectedNode ? (
            <div className="mt-4 space-y-4 text-sm">
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-primary">Label</p>
                <p className="mt-1 font-semibold text-foreground">{selectedNode.label}</p>
              </div>

              <div className="grid gap-2">
                <div className="flex items-center justify-between text-xs uppercase tracking-wide text-muted-foreground">
                  <span>Type</span>
                  <span className="rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 font-medium capitalize text-primary">
                    {selectedNode.type}
                  </span>
                </div>
                <div className="text-xs text-muted-foreground">
                  {expandedNodes[selectedNode.id]
                    ? "Neighborhood loaded"
                    : "Click expand to load neighbors"}
                </div>
              </div>

              {selectedMetadataEntries.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Metadata</p>
                  <dl className="grid gap-1 text-xs text-muted-foreground">
                    {selectedMetadataEntries.map(([key, value]) => (
                      <div key={key} className="flex justify-between gap-2">
                        <dt className="font-medium capitalize text-foreground/70">{key}</dt>
                        <dd className="text-right">{formatMetadataValue(value)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ) : null}

              {connectedNodes.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Connections</p>
                  <ul className="space-y-1">
                    {connectedNodes.map((node) => (
                      <li
                        key={node.id}
                        className="flex items-center justify-between rounded-md border border-border/60 bg-muted/30 px-2 py-1 text-xs text-muted-foreground"
                      >
                        <span className="truncate font-medium text-foreground/80">{node.label}</span>
                        <span className="rounded-full bg-background px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                          {node.type}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">
                  Select a node to inspect its metadata and relationships.
                </p>
              )}
            </div>
          ) : (
            <p className="mt-3 text-sm text-muted-foreground">
              Select a node in the graph to view its metadata and connections.
            </p>
          )}
        </div>

        <div className="rounded-lg border bg-card p-5 text-sm text-muted-foreground">
          <h3 className="text-base font-semibold text-foreground">How to explore</h3>
          <ul className="mt-3 space-y-2 text-sm">
            <li>Click any node to highlight it and load its neighborhood.</li>
            <li>Use the action buttons to reset the graph or force a fresh expansion.</li>
            <li>
              Concept-to-concept edges are highlighted in amber to differentiate them from paper associations.
            </li>
          </ul>
        </div>
      </aside>
    </div>
  );
};

export default GraphExplorer;

