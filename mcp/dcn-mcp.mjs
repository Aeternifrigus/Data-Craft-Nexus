#!/usr/bin/env node
// Runs the Data Craft Nexus MCP server over stdio, the way coding agents
// (Claude Code, Claude Desktop, Cursor and others) start local servers.
//
//   node mcp/dcn-mcp.mjs
//
// stdout carries the protocol, so anything else goes to stderr.

import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { createServer } from './server.mjs';

const server = createServer();
await server.connect(new StdioServerTransport());
console.error('data-craft-nexus MCP server running on stdio');
