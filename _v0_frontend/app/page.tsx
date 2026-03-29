import { FlowmindIDE } from "../components/local-flowmind/flowmind-ide";

export default function Page() {
  return (
    <FlowmindIDE 
      config={{
        title: "AGENT SWARM 2.0",
        initialMessage: "Welcome to AgentSwarm 2.0. Ready for architectural upgrades."
      }} 
    />
  );
}
