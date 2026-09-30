import type { TableData } from "../api";

export function ResultTable({ table }: { table: TableData }) {
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            {table.columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, index) => (
            <tr key={index}>
              {row.map((cell, position) => (
                <td key={position}>{cell === null ? "—" : String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
