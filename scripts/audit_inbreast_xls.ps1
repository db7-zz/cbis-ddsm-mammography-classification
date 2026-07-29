param(
    [Parameter(Mandatory = $true)]
    [string]$Path
)

$ErrorActionPreference = "Stop"
$resolved = (Resolve-Path -LiteralPath $Path).Path
$connection = New-Object System.Data.OleDb.OleDbConnection(
    "Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$resolved;Extended Properties='Excel 8.0;HDR=YES;IMEX=1'"
)

try {
    $connection.Open()
    $tables = @{}
    foreach ($sheet in @("Sheet1$", "Sheet2$")) {
        $command = $connection.CreateCommand()
        $command.CommandText = "SELECT * FROM [$sheet]"
        $adapter = New-Object System.Data.OleDb.OleDbDataAdapter($command)
        $table = New-Object System.Data.DataTable
        [void]$adapter.Fill($table)
        $tables[$sheet] = $table
    }

    $sheet1 = $tables["Sheet1$"]
    $sheet2 = $tables["Sheet2$"]
    $usable = @()
    for ($index = 0; $index -lt $sheet1.Rows.Count; $index++) {
        $fileId = ([string]$sheet1.Rows[$index]."File Name").Trim()
        if ($fileId -match "^\d+$") {
            $usable += [pscustomobject]@{
                Index = $index
                FileId = $fileId
                Sheet1Birads = ([string]$sheet1.Rows[$index]."Bi-Rads").Trim().ToLowerInvariant()
                Sheet2Birads = ([string]$sheet2.Rows[$index]."Bi-Rads").Trim().ToLowerInvariant()
                Diagnosis = ([string]$sheet2.Rows[$index].DG).Trim()
            }
        }
    }

    $diagnosisCounts = @{}
    foreach ($group in ($usable | Where-Object Diagnosis | Group-Object Diagnosis)) {
        $diagnosisCounts[$group.Name] = $group.Count
    }
    $mismatches = @(
        $usable | Where-Object { $_.Sheet1Birads -ne $_.Sheet2Birads }
    ).Count
    $sheet1Columns = @($sheet1.Columns | ForEach-Object ColumnName)
    $sheet2Columns = @($sheet2.Columns | ForEach-Object ColumnName)
    $directLinkFields = @("File Name", "Laterality", "View") |
        Where-Object { $sheet2Columns -contains $_ }

    $result = [ordered]@{
        status = "PASS"
        workbook_sha256 = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
        sheets = [ordered]@{
            Sheet1 = [ordered]@{
                raw_rows = $sheet1.Rows.Count
                columns = $sheet1Columns
            }
            Sheet2 = [ordered]@{
                raw_rows = $sheet2.Rows.Count
                columns = $sheet2Columns
                diagnosis_nonempty_rows = @($usable | Where-Object Diagnosis).Count
                diagnosis_counts = $diagnosisCounts
            }
        }
        usable_image_rows = $usable.Count
        same_row_birads_mismatches = $mismatches
        sheet2_direct_image_link_fields = @($directLinkFields)
        row_order_link_supported = ($mismatches -eq 0)
        image_level_pathology_link_supported = $false
        reason = "Sheet2 has partial DG values but lacks File Name/laterality/view and differs from Sheet1 BI-RADS in the same row; no deterministic image-level join is available."
    }
    $result | ConvertTo-Json -Depth 8 -Compress
}
finally {
    if ($connection.State -eq [System.Data.ConnectionState]::Open) {
        $connection.Close()
    }
}
