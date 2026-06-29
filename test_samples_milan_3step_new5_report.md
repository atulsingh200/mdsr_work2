# Bi-Encoder Sorting Results — 5 Three-Step Workflows

**Model:** BGE-small dual-encoder (`runs/classifier/best_mlp_bge_small_aep`)
**Overall accuracy: 2 / 5**

---

## Workflow 1 — Schema → Dataset → Data Ingestion ✅ CORRECT

| Step | Description |
|------|-------------|
| t1 | Define XDM schema (choose base class, add field groups, enable for Profile) |
| t2 | Create dataset from that schema (empty container, toggle Profile on) |
| t3 | Ingest CSV via Sources connector, map columns to XDM fields, verify batch |

**Given order:** t1 → t2 → t3
**Predicted order:** t1 → t2 → t3

---

## Workflow 2 — Segment Definition → Audience Activation → Destination Mapping ✅ CORRECT

| Step | Description |
|------|-------------|
| t1 | Build audience segment in Segment Builder (set rules, save) |
| t2 | Activate segment to a destination (pick destination, set schedule) |
| t3 | Map profile attributes to destination fields (add field mappings, click Finish) |

**Given order:** t1 → t2 → t3
**Predicted order:** t1 → t2 → t3

---

## Workflow 3 — Identity Namespace → Identity Graph → Profile Merge ❌ WRONG

| Step | Description |
|------|-------------|
| t1 | Create custom identity namespace (display name, symbol, identity type) |
| t2 | Inspect identity graph stitching (nodes = namespaces, edges = associations) |
| t3 | Configure merge policy (identity stitching option, attribute merge method) |

**Given order:** t1 → t2 → t3
**Predicted order:** t2 → t1 → t3

---

## Workflow 4 — Datastream → Event Forwarding → Reporting Dataset ❌ WRONG

| Step | Description |
|------|-------------|
| t1 | Configure Web SDK datastream (schema, services, generates datastream ID) |
| t2 | Set up event forwarding rules (trigger condition, enrich/route on edge) |
| t3 | Verify data landed in reporting dataset (check batch activity, Profile Viewer) |

**Given order:** t1 → t2 → t3
**Predicted order:** t2 → t3 → t1

---

## Workflow 5 — Journey Trigger → Journey Canvas Design → Journey Publishing ❌ WRONG

| Step | Description |
|------|-------------|
| t1 | Configure entry event trigger (unitary or segment qualification, save event) |
| t2 | Design journey canvas (add Wait, Email, Condition nodes, connect with arrows) |
| t3 | Publish journey (validation sweep, status → Live, monitor report tab) |

**Given order:** t1 → t2 → t3
**Predicted order:** t2 → t1 → t3

---

## Summary

| # | Workflow | Given | Predicted | Result |
|---|----------|-------|-----------|--------|
| 1 | Schema → Dataset → Data Ingestion | t1 → t2 → t3 | t1 → t2 → t3 | ✅ |
| 2 | Segment Definition → Activation → Mapping | t1 → t2 → t3 | t1 → t2 → t3 | ✅ |
| 3 | Identity Namespace → Graph → Merge | t1 → t2 → t3 | t2 → t1 → t3 | ❌ |
| 4 | Datastream → Event Forwarding → Dataset | t1 → t2 → t3 | t2 → t3 → t1 | ❌ |
| 5 | Journey Trigger → Canvas → Publishing | t1 → t2 → t3 | t2 → t1 → t3 | ❌ |

**Sorting accuracy: 2 / 5 (40%)**
